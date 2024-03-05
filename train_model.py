import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import wandb
from tqdm.auto import tqdm

def train(feature_reconstructor,feature_mask,factorVAE,env_factorVAE, train_dataloader, featrue_optimizer, optimizer, env_optimizer,featrue_scheduler,scheduler,env_scheduler, wandb,args,epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_mask.to(device)
    feature_reconstructor.to(device)
    feature_reconstructor.train()
    factorVAE.to(device)
    factorVAE.train()
    env_factorVAE.to(device)
    env_factorVAE.train()
    total_loss = 0
    total_env_loss = 0
    total_diff_loss = 0
    total_recon_loss = 0
    total_env_recon_loss = 0
    total_self_recon_loss = 0
    total_recon_diff_loss = 0
    total_kl_diff_loss = 0
    total_rank_loss = 0
    total_env_rank_loss = 0
    total_kl_loss = 0
    total_env_kl_loss = 0
    total_rank_diff_loss = 0
    path = epoch %  3
    with tqdm(total=len(train_dataloader)) as pbar:
        for char, returns in train_dataloader:
            if char.shape[1] != args.seq_len:
                continue
            inputs = char.to(device)
            labels = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            labels = labels.float()

            mask = feature_mask(inputs[...,:-args.env_size])[...,0]
            # mask = mask.unsqueeze(-1)
            new_features = mask * inputs[...,:-args.env_size]
            new_features_env = torch.cat([new_features,inputs[...,-args.env_size:]],dim=-1)
            self_recondstruction = feature_reconstructor(new_features)
            if torch.isnan(torch.sum(new_features)):
                print(epoch)
            # std = torch.exp(0.5 * log_var)
            # eps = torch.randn_like(std)
            # stock_latent = eps * std + mu
            env = inputs[:,:,-args.env_size:]
            batch_size = inputs.shape[0]
            loss, recon_loss,rank_loss, kl_loss,reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factorVAE(new_features, labels)
            env_loss, env_recon_loss,env_rank_loss,env_kl_loss,env_reconstruction, env_factor_mu, env_factor_sigma, env_pred_mu, env_pred_sigma = env_factorVAE(new_features_env,labels)

            if path == 0:
                #feature extractor
                ranks = torch.ones_like(labels)
                ranks_index = torch.argsort(labels,dim=0)
                ranks[ranks_index,0] = torch.arange(0,batch_size).reshape(ranks.shape).float().to(device)
                ranks = (ranks - torch.mean(ranks))/torch.std(ranks)
                ranks = ranks ** 2
                all_ones = torch.ones(batch_size,1).to(device)
                pre_dif =  (torch.matmul(reconstruction, torch.transpose(all_ones, 0, 1)) 
                                - torch.matmul(all_ones, torch.transpose(reconstruction, 0, 1)))
                env_pre_dif = (
                        torch.matmul(all_ones, torch.transpose(env_reconstruction,0,1)) -
                        torch.matmul(env_reconstruction, torch.transpose(all_ones, 0,1))
                    )
                rank_diff_loss = torch.mean(ranks * F.relu(pre_dif*env_pre_dif))
                
                featrue_optimizer.zero_grad()
                self_recon_loss = F.mse_loss(inputs[...,:-args.env_size],self_recondstruction)
                recon_diff_loss = F.mse_loss(reconstruction,env_reconstruction)
                kl_diff_loss = KL_Divergence(pred_mu,pred_sigma,env_pred_mu,env_pred_sigma)
                diff_loss = self_recon_loss + recon_diff_loss + kl_diff_loss+rank_diff_loss
                total_diff_loss += diff_loss.item()* inputs.size(0)
                total_rank_diff_loss += rank_diff_loss.item()* inputs.size(0)
                total_self_recon_loss += self_recon_loss.item()* inputs.size(0)
                total_recon_diff_loss += recon_diff_loss.item()* inputs.size(0)
                total_kl_diff_loss += kl_diff_loss.item()* inputs.size(0)
                # print(diff_loss.item())
                diff_loss.backward()
                # total_grad_sum = 0.0
                # for param in feature_mask.parameters():
                #     if param.grad is not None:
                #         total_grad_sum += param.grad.data.sum()

                # print("Total sum of gradients:", total_grad_sum)
                # if torch.isnan(total_grad_sum):
                #     print("stop")
                # torch.nn.utils.clip_grad_norm_(list(feature_extractor.parameters())+list(feature_mask.parameters()), max_norm=5)
                featrue_optimizer.step()
                featrue_scheduler.step()
                # for name,i in feature_mask.named_parameters():
                #     print(name,torch.sum(i),i.grad)
            elif path in [1]:
                # without env
                optimizer.zero_grad()

                total_loss += loss.item()* inputs.size(0)
                total_recon_loss += recon_loss.item() * inputs.size(0)
                total_kl_loss += kl_loss.item() * inputs.size(0)
                total_rank_loss += rank_loss.item() * inputs.size(0)
                loss.backward()
                optimizer.step()
                scheduler.step()
            elif path in [2]:
                # with env
                env_optimizer.zero_grad()
                total_env_loss += env_loss.item()* inputs.size(0)
                total_env_recon_loss += env_recon_loss.item() * inputs.size(0)
                total_env_rank_loss += env_rank_loss.item() * inputs.size(0)
                total_env_kl_loss += env_kl_loss.item() * inputs.size(0)
                env_loss.backward()
                env_optimizer.step()
                env_scheduler.step()

            pbar.update(1)
        # print(loss)
    avg_loss = total_loss / len(train_dataloader.dataset)
    env_avg_loss = total_env_loss / len(train_dataloader.dataset)
    diff_avg_loss = total_diff_loss / len(train_dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(train_dataloader.dataset)
    avg_env_recon_loss = total_env_recon_loss / len(train_dataloader.dataset)
    avg_self_recon_loss = total_self_recon_loss / len(train_dataloader.dataset)
    avg_recon_diff_loss = total_recon_diff_loss / len(train_dataloader.dataset)
    avg_kl_diff_loss = total_kl_diff_loss / len(train_dataloader.dataset)
    avg_rank_loss = total_rank_loss / len(train_dataloader.dataset)
    avg_env_rank_loss = total_env_rank_loss / len(train_dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(train_dataloader.dataset)
    avg_env_kl_loss = total_env_kl_loss / len(train_dataloader.dataset)
    avg_rank_diff_loss = total_rank_diff_loss / len(train_dataloader.dataset)
    return avg_loss,avg_recon_loss,env_avg_loss,avg_env_recon_loss,diff_avg_loss,avg_self_recon_loss,avg_recon_diff_loss,avg_kl_diff_loss,avg_rank_loss,avg_env_rank_loss,avg_kl_loss,avg_env_kl_loss,avg_rank_diff_loss

@torch.no_grad()
def validate(feature_extractor,feature_mask,factorVAE,env_factorVAE, dataloader, featrue_optimizer, optimizer,env_optimizer,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factorVAE.to(device)
    factorVAE.eval()
    total_loss = 0
    recon_loss = 0
    total_rank_loss = 0
    total_env_kl_loss = 0 
    total_rankic = 0   
    with tqdm(total=len(dataloader)) as pbar:
        for char, returns in dataloader:
            if char.shape[1] != args.seq_len:
                continue
            inputs = char.to(device)
            labels = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            labels = labels.float()
            batch_size = inputs.shape[0]
            mask = feature_mask(inputs[...,:-args.env_size])[...,0]
            # mask = mask.unsqueeze(-1)
            new_features = mask * inputs[...,:-args.env_size]
            # self_recondstruction = feature_reconstructor(new_features)
            env = inputs[...,-args.env_size:]
            loss, reconstruction_loss,rank_loss,env_kl_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factorVAE(new_features, labels)
            total_loss += loss.item() * inputs.size(0)
            recon_loss += reconstruction_loss.item() * inputs.size(0)
            total_rank_loss += rank_loss.item() * inputs.size(0)
            total_env_kl_loss += env_kl_loss.item() * inputs.size(0)
            
            
            ranks = pred_ranks = torch.ones_like(labels)
            ranks_index = torch.argsort(labels,dim=0)
            ranks[ranks_index,0] = torch.arange(0,batch_size).reshape(ranks.shape).float().to(device)
            ranks = (ranks - torch.mean(ranks))/torch.std(ranks)
            
            
            pred_ranks_index = torch.argsort(reconstruction,dim=0)
            pred_ranks[pred_ranks_index,0] = torch.arange(0,batch_size).reshape(ranks.shape).float().to(device)
            pred_ranks = (pred_ranks - torch.mean(pred_ranks))/torch.std(pred_ranks)
            
            rankic = (ranks * pred_ranks).mean()
            total_rankic += rankic.item()*inputs.size(0)
            
            pbar.update(1)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = recon_loss / len(dataloader.dataset)
    avg_rank_loss = total_rank_loss / len(dataloader.dataset)
    avg_env_kl_loss = total_env_kl_loss / len(dataloader.dataset)
    avg_rankic = total_rankic/len(dataloader.dataset)
    
    return avg_loss,avg_recon_loss,avg_rank_loss,avg_env_kl_loss,avg_rankic

# @torch.no_grad()
# def test(feature_extractor,factorVAE,env_factorVAE, dataloader, featrue_optimizer, optimizer,env_optimizer,args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     factorVAE.to(device)
#     factorVAE.eval()
#     total_loss = 0
#     with tqdm(total=len(dataloader)-args.seq_len+1) as pbar:
#         for char, returns in dataloader:
#             if char.shape[1] != args.seq_len:
#                 continue
#             inputs = char.to(device)
#             labels = returns[:,-1].reshape(-1,1).to(device)
#             inputs = inputs.float()
#             labels = labels.float()
#             stock_latent = feature_extractor(inputs[...,:-12])
#             env = inputs[...,-12:]
#             loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factorVAE(stock_latent, labels)
#             total_loss += loss.item() * inputs.size(0)
#             pbar.update(1)
#     avg_loss = total_loss / len(dataloader.dataset)
#     return avg_loss

def run(factor_model, train_loader, val_loader, test_loader, lr, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # wandb.init(project="FactorVAE", name="replicate")
    factor_model.to(device)
    best_val_loss = float('inf')
    optimizer = torch.optim.AdamW(factor_model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_loss = train(factor_model, train_loader, optimizer)
        val_loss = validate(factor_model, val_loader)
        test_loss = test(factor_model, test_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(factor_model.state_dict(), 'best_model.pt')
def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

def KL_Divergence(mu1, sigma1, mu2, sigma2):
    #! mu1, mu2: (batch_size, 1)
    #! sigma1, sigma2: (batch_size, 1)
    #! output: (batch_size, 1)
    kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
    return kl_div