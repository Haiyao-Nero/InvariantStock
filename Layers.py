
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

class FeatureReconstructor(nn.Module):
    def __init__(self, feat_dim):
        super(FeatureReconstructor, self).__init__()
        self.decoder= nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, feat_dim))
    def forward(self,x):
        return self.decoder(x)


class FeatureExtractor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(feat_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):

        out = self.linear(x)
        out = self.leakyrelu(out)

        stock_latent, _ = self.gru(out)
        return stock_latent[:,-1,:]
    
class FeatureMask(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers=1):
        super(FeatureMask, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.normalize = nn.LayerNorm(feat_dim)
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(feat_dim, hidden_dim, num_layers, batch_first=True)
        self.generator_fc = nn.Linear(hidden_dim,feat_dim*2)
        self.softmax = nn.Softmax(dim=-1)

        
    def _independent_straight_through_sampling(self, z):
        z_hard = torch.eq(z,torch.max(z,-1).values.unsqueeze(-1)).to(z.dtype)
        z = (z_hard-z).detach()+z
        return z

    def forward(self, x):
        batch,length,num_feat = x.shape
        out = self.linear(x)
        out = self.leakyrelu(out)
        stock_latent, _ = self.gru(out)
        z = self.generator_fc(stock_latent)
        z = self.softmax(z.reshape(batch,length,num_feat,2))
        return self._independent_straight_through_sampling(z)#* stock_latent[-1]: (batch_size, hidden_dim)

class FactorEncoder(nn.Module):
    def __init__(self, factor_dims, num_portfolio, hidden_dim):
        super(FactorEncoder, self).__init__()
        self.factor_dims = factor_dims
        self.linear = nn.Linear(hidden_dim, num_portfolio)
        self.softmax = nn.Softmax(dim=1)
        
        self.linear2 = nn.Linear(num_portfolio, factor_dims)
        self.softplus = nn.Softplus()

    def mapping_layer(self, portfolio_return):
        mean = self.linear2(portfolio_return.squeeze(1))
        sigma = self.softplus(mean)
        return mean, sigma
    
    def forward(self, stock_latent, returns):
        weights = self.linear(stock_latent)
        weights = self.softmax(weights)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        portfolio_return = torch.mm(weights.transpose(1,0), returns) 
        
        return self.mapping_layer(portfolio_return)

class AlphaLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AlphaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_dim, 1)
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, stock_latent):
        stock_latent = self.linear1(stock_latent)
        stock_latent = self.leakyrelu(stock_latent)
        alpha_mu = self.mu_layer(stock_latent)
        alpha_sigma = self.sigma_layer(stock_latent)
        return alpha_mu, self.softplus(alpha_sigma)
        
class BetaLayer(nn.Module):
    """calcuate factor exposure beta(N*K)"""
    def __init__(self, hidden_dim, factor_dims):
        super(BetaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, factor_dims)
    
    def forward(self, stock_latent):
        beta = self.linear1(stock_latent)
        return beta
        
class FactorDecoder(nn.Module):
    def __init__(self, alpha_layer, beta_layer):
        super(FactorDecoder, self).__init__()

        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, stock_latent, factor_mu, factor_sigma):
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)

        factor_sigma[factor_sigma == 0] = 1e-6
        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2) + 1e-6)
        return self.reparameterize(mu, sigma)
    
    def predict(self, stock_latent, factor_mu):
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)
        factor_mu = factor_mu.view(-1, 1)
        mu = alpha_mu + torch.matmul(beta, factor_mu)
        return mu

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, stock_latent):

        self.key = self.key_layer(stock_latent)
        self.value = self.value_layer(stock_latent)
        
        attention_weights = torch.matmul(self.query, self.key.transpose(1,0)) 
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.key.shape[0])+ 1e-6)
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights) 
        attention_weights = F.softmax(attention_weights, dim=0) 
        
        #! calculate context vector
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return torch.zeros_like(self.value[0])
        else:
            context_vector = torch.matmul(attention_weights, self.value) # (H)
            return context_vector 

class FatorPrior(nn.Module):
    def __init__(self, batch_size, hidden_dim, factor_dim):
        super(FatorPrior, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.factor_dim = factor_dim
        self.attention_layers = nn.ModuleList([AttentionLayer(self.hidden_dim) for _ in range(factor_dim)])
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_dim, 1)
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        
        for i in range(self.factor_dim):
            attention_layer = self.attention_layers[i](stock_latent)
            if i == 0:
                h_multi = attention_layer
            else:
                h_multi = torch.cat((h_multi, attention_layer), dim=0)
        h_multi = h_multi.view(self.factor_dim, -1)

        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        pred_mu = self.mu_layer(h_multi)
        pred_sigma = self.sigma_layer(h_multi)
        pred_sigma = self.softplus(pred_sigma)
        pred_mu = pred_mu.view(-1)
        pred_sigma = pred_sigma.view(-1)
        return pred_mu, pred_sigma

 
class Predictor(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_prior_model,args):
        super(Predictor, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_prior_model = factor_prior_model
        self.device = args.device

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div

    def forward(self, x, returns):
        batch_size, seq_length, feat_dim = x.shape
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_prior_model(stock_latent)

        ranks = torch.ones_like(returns)
        ranks_index = torch.argsort(returns,dim=0)
        ranks[ranks_index,0] = torch.arange(0,batch_size).reshape(ranks.shape).float().to(self.device)
        ranks = (ranks-torch.mean(ranks))/torch.std(ranks)
        ranks = ranks ** 2

        reconstruction_loss = F.mse_loss(ranks*reconstruction, ranks*returns)
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)
        
        all_ones = torch.ones(batch_size,1).to(self.device)
        pre_pw_dif =  (torch.matmul(reconstruction, torch.transpose(all_ones, 0, 1)) 
                        - torch.matmul(all_ones, torch.transpose(reconstruction, 0, 1)))
        gt_pw_dif = (
                torch.matmul(all_ones, torch.transpose(returns,0,1)) -
                torch.matmul(returns, torch.transpose(all_ones, 0,1))
            )
        
        rank_loss = torch.mean(ranks * F.relu(pre_pw_dif*gt_pw_dif))
        vae_loss = reconstruction_loss + kl_divergence + rank_loss
        return vae_loss, reconstruction_loss,rank_loss,kl_divergence,reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma #! reconstruction, factor_mu, factor_sigma

    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, _ = self.factor_prior_model(stock_latent)
        y_pred = self.factor_decoder.predict(stock_latent, pred_mu)
        return y_pred
