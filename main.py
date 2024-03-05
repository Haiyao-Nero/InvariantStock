import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm
import argparse
from Layers import FeatureReconstructor,EnvFeatureExtractor,EnvFactorVAE, FactorVAE,FeatureMask, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from dataset import StockDataset,DynamicBatchSampler
from train_model import train, validate
from utils import set_seed, DataArgument
import wandb

parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--num_latent', type=int, default=20, help='number of latent variables')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--num_factor', type=int, default=10, help='number of factors')
parser.add_argument('--hidden_size', type=int, default=20, help='hidden size')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--run_name', type=str, default="RationalStock_yearmonthenv", help='name of the run')
parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
parser.add_argument('--normalize', action='store_true', help='whether to normalize the data')
parser.add_argument('--env_size', type=int, default=40, help='env dimension')
parser.add_argument('--device', default="cuda:0",type=str, help='devices')

args = parser.parse_args()

data_args = DataArgument(use_qlib=False, normalize=True, select_feature=False)
args.save_dir = args.save_dir+"/"+str(args.num_factor)
assert args.seq_len == data_args.seq_len, "seq_len in args and data_args must be the same"
assert args.normalize == data_args.normalize, "normalize in args and data_args must be the same"
        
# if args.normalize:
#     print("*************** Use normalized data ***************")
#     print("select_feature:", data_args.select_feature)
#     train_df = pd.read_pickle(f"{data_args.save_dir}/train_all.pkl")
#     valid_df = pd.read_pickle(f"{data_args.save_dir}/valid_all.pkl")
#     test_df = pd.read_pickle(f"{data_args.save_dir}/test_all.pkl")
    
# else:
#     print("Use raw data")
train_df = pd.read_pickle('./data/train_all.pkl')
valid_df = pd.read_pickle('./data/valid_all.pkl')
# test_df = pd.read_pickle('./data/test_all.pkl')
    
train_index = np.load(f"{data_args.save_dir}/train_index.npy")
valid_index = np.load(f"{data_args.save_dir}/valid_index.npy")
# index_columns = ['sh000001open', 'sh000001high', 'sh000001low', 'sh000001close',
#        'sh000001volume', 'sh000001next_open', 'sh000698open', 'sh000698high',
#        'sh000698low', 'sh000698close', 'sh000698volume', 'sh000698next_open',
#        'sz399001open', 'sz399001high', 'sz399001low', 'sz399001close',
#        'sz399001volume', 'sz399001next_open', 'sz399006open', 'sz399006high',
#        'sz399006low', 'sz399006close', 'sz399006volume', 'sz399006next_open']
month = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",]
year = [str(i) for i in range(1990,2025)]
args.env_size = len(month+year)
# drop_columns = month+year
# drop_columns = index_columns
# train_df.drop(columns=drop_columns,inplace=True)
# valid_df.drop(columns=drop_columns,inplace=True)


args.num_latent = len(train_df.columns)-args.env_size-1
if args.wandb:
    wandb.init(project="FactorVAE_US", config=args, name=f"{args.run_name}")
    wandb.config.update(args)
    # wandb.log({"train_df": train_df, "valid_df": valid_df, "test_df": test_df})

def main(args, data_args):

    set_seed(args.seed)
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create model
    feature_reconstructor = FeatureReconstructor(num_latent=args.num_latent)
    feature_mask = FeatureMask(num_latent=args.num_latent, hidden_size=args.num_latent)
    
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.batch_size, args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor,args)

    env_feature_extractor = EnvFeatureExtractor(num_latent=args.num_latent+args.env_size, hidden_size=args.hidden_size)
    env_factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_latent, hidden_size=args.hidden_size)
    env_alpha_layer = AlphaLayer(args.hidden_size)
    env_beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    env_factor_decoder = FactorDecoder(env_alpha_layer, env_beta_layer)
    env_factor_predictor = FactorPredictor(args.batch_size, args.hidden_size, args.num_factor)
    env_factorVAE = EnvFactorVAE(env_feature_extractor, env_factor_encoder, env_factor_decoder, env_factor_predictor,args)

    # create dataloaders
    # Assuming you want to create a mini-batch of size 300
    train_ds = StockDataset(train_df,train_index, args.batch_size, args.seq_len)
    valid_ds = StockDataset(valid_df,valid_index, args.batch_size, args.seq_len)
    #test_ds = StockDataset(test_df, args.batch_size, args.seq_len)
    def collate_fn(batch):
        data, label = zip(*batch)
        return torch.stack(data),torch.stack(label)
    
    train_batch_sizes = pd.DataFrame([ i[0] for i in train_df.index[train_index[:,0]].values]).value_counts(sort=False).values
    train_batch_sampler = DynamicBatchSampler(train_ds,train_batch_sizes)
    
    valid_batch_sizes = pd.DataFrame([ i[0] for i in valid_df.index[valid_index[:,0]].values]).value_counts(sort=False).values
    valid_batch_sampler = DynamicBatchSampler(valid_ds,valid_batch_sizes)



    train_dataloader = DataLoader(train_ds, batch_sampler=train_batch_sampler, shuffle=False, num_workers=4)
    valid_dataloader = DataLoader(valid_ds, batch_sampler=valid_batch_sampler, shuffle=False, num_workers=4)
    #test_dataloader = DataLoader(test_ds, batch_size=300, shuffle=False, num_workers=4)

    # device = args.device
    device = args.device

    factorVAE.to(device)
    best_rankic = 0

    featrue_optimizer = torch.optim.Adam(list(feature_reconstructor.parameters())+list(feature_mask.parameters()), lr=args.lr)
    featrue_scheduler = torch.optim.lr_scheduler.OneCycleLR(featrue_optimizer,pct_start=0.1, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs//3)
    # featrue_scheduler = torch.optim.lr_scheduler.StepLR(featrue_optimizer, step_size=len(train_dataloader), gamma=0.9, last_epoch=-1, verbose=False)
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,pct_start=0.1, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs//3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader), gamma=0.9, last_epoch=-1, verbose=False)
    env_optimizer = torch.optim.Adam(env_factorVAE.parameters(), lr=args.lr)
    env_scheduler = torch.optim.lr_scheduler.OneCycleLR(env_optimizer, max_lr=args.lr,pct_start=0.1, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs//3)
    # env_scheduler = torch.optim.lr_scheduler.StepLR(env_optimizer, step_size=len(train_dataloader), gamma=0.9, last_epoch=-1, verbose=False)

    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss,recon_loss,env_loss,env_recon_loss,diff_loss,self_recon_loss,recon_diff_loss,kl_diff_loss,rank_loss,env_rank_loss,kl_loss,env_kl_loss,rank_diff_loss = train(feature_reconstructor,feature_mask,factorVAE,env_factorVAE, train_dataloader, featrue_optimizer, optimizer,env_optimizer,featrue_scheduler,scheduler,env_scheduler,wandb,args,epoch=epoch)
        val_loss,val_recon_loss,val_rank_loss,val_kl_loss,avg_rankic = validate(feature_reconstructor,feature_mask,factorVAE,env_factorVAE, valid_dataloader, featrue_optimizer, optimizer,env_optimizer,args)
        test_loss = np.NaN #test(factorVAE, test_dataloader, args)
        path = path = epoch % 3
        # wandb.log({"Reconstruction Loss": reconstruction_loss})
        print(f"Epoch {epoch+1}: ",{"Validation Toal Loss": round(val_loss,6),"Validation Recon Loss": round(val_recon_loss,6),"Validation Ranking Loss": round(val_rank_loss,6),"Validation KL Loss": round(val_kl_loss,6),"Validation RankIC":round(avg_rankic,6)})
        if args.wandb:
            wandb.log({"Validation Toal Loss": val_loss,"Validation Recon Loss": val_recon_loss,"Validation Ranking Loss": val_rank_loss,"Validation KL Loss": val_kl_loss,"Validation RankIC":avg_rankic}, step=epoch)
            if path == 0:
                wandb.log({"Different Loss": diff_loss,"Self Reconstruction Loss":self_recon_loss,"Reconstruction Diff Loss":recon_diff_loss,"KL Diff Loss":kl_diff_loss,"Rank Diff Loss":rank_diff_loss}, step=epoch)
            elif path in [1]:
                wandb.log({"No Env Loss": train_loss,"No Env Recon Loss": recon_loss,"No Env Ranking Loss": rank_loss,"No Env KL Loss": kl_loss}, step=epoch)
            elif path in [2]:
                wandb.log({"With Env Loss": env_loss,"With Env Recon Loss": env_recon_loss,"With Env Ranking Loss": env_rank_loss,"With Env KL Loss": env_kl_loss}, step=epoch)

        if path == 0:
            print(f"Epoch {epoch+1}: ",{"Different Loss": round(diff_loss,6),"Self Reconstruction Loss":round(self_recon_loss,6),"Reconstruction Diff Loss":round(recon_diff_loss,6),"KL Diff Loss":round(kl_diff_loss,6),"Rank Diff Loss":round(rank_diff_loss,6)})
        elif path in [1]:
            print(f"Epoch {epoch+1}: ",{"No Env Loss": round(train_loss,6),"No Env Recon Loss": round(recon_loss,6),"No Env Ranking Loss": round(rank_loss,6),"No Env KL Loss": round(kl_loss,6)})
        elif path in [2]:
            print(f"Epoch {epoch+1}: ",{"With Env Loss": round(env_loss,6),"With Env Recon Loss": round(env_recon_loss,6),"With Env Ranking Loss": round(env_rank_loss,6),"With Env KL Loss": round(env_kl_loss,6)})
        # print(f"Epoch {epoch+1}: Train Total Loss: {train_loss:.6f},Train Recon Loss: {recon_loss:.6f}, Validation Total Loss: {val_loss:.6f}, Validation Recon Loss: {val_recon_loss:.6f},learning rate: {scheduler.get_lr()}") #Test Loss: {test_loss:.6f},
        if avg_rankic > best_rankic:
            best_rankic = avg_rankic
            #? save model in save_dir

            #? torch.save
            factorVAE_root = os.path.join(args.save_dir, f'best_factorVAE_{args.run_name}_{args.num_factor}.pt')
            factorMask_root = os.path.join(args.save_dir, f'best_factorMask_{args.run_name}_{args.num_factor}.pt')
            torch.save(factorVAE.state_dict(), factorVAE_root)
            torch.save(feature_mask.state_dict(), factorMask_root)
            # torch.save(feature_extractor.state_dict(), factorExtractor_root)
    factorVAE_root = os.path.join(args.save_dir, f'last_factorVAE_{args.run_name}_{args.num_factor}.pt')
    factorMask_root = os.path.join(args.save_dir, f'last_factorMask_{args.run_name}_{args.num_factor}.pt')
    torch.save(factorVAE.state_dict(), factorVAE_root)
    torch.save(feature_mask.state_dict(), factorMask_root)

    if args.wandb:
        wandb.log({"Best Validation Loss": best_rankic})
        wandb.finish()

if __name__ == '__main__':
    main(args, data_args)
