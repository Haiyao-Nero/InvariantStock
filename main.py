import multiprocessing
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import argparse
from Layers import FeatureReconstructor, Predictor,FeatureMask, FeatureExtractor, FactorDecoder, FactorEncoder, FatorPrior, AlphaLayer, BetaLayer
from dataset import StockDataset,DynamicBatchSampler
from train_model import train, validate
from utils import set_seed, DataArgument,generate_prediction_scores
import wandb

parser = argparse.ArgumentParser(description='Train a predictor model on stock data')

parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--feat_dim', type=int, default=20, help='features dimension')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--factor_dim', type=int, default=10, help='number of factors')
parser.add_argument('--hidden_dim', type=int, default=20, help='hidden variables dimension')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--run_name', type=str, default="InvariantStock", help='name of the run')
parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
parser.add_argument('--normalize', action='store_true', help='whether to normalize the data')
parser.add_argument('--device', default="cuda:0",type=str, help='devices')
args = parser.parse_args()

def rankic(df):
    ic = df.groupby('datetime').apply(lambda df: df["label"].corr(df["pred"]))
    ric = df.groupby('datetime').apply(lambda df: df["label"].corr(df["pred"], method="spearman"))
    
    print({
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
        })
    return ric



data_args = DataArgument(use_qlib=False, normalize=True, select_feature=False)
args.save_dir = args.save_dir+"/"+str(args.factor_dim)
        
dataset = pd.read_pickle(f"{data_args.save_dir}/usdataset_norm.pkl")



train_index = np.load(f"{data_args.save_dir}/train_index.npy")
valid_index = np.load(f"{data_args.save_dir}/valid_index.npy")
test_index = np.load(f"{data_args.save_dir}/test_index.npy")

args.feat_dim = len(dataset.columns)-1
if args.wandb:
    wandb.init(project="InvariantStock", config=args, name=f"{args.run_name}")
    wandb.config.update(args)

def add_env(date,df):
    month = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",]
    begin_year = 2024
    end_year = 2024
    year = [str(i) for i in range(begin_year,end_year+1)]
    df[month]=0
    df[year]=0
    df.loc[df.index.get_level_values("datetime") == date,month[date.month-1]]=1
    df.loc[df.index.get_level_values("datetime") == date,year[date.year-begin_year]]=1
    return df
def multi_add_env(dataset):
    pool = multiprocessing.Pool()
    results = pool.starmap(add_env, [*dataset.groupby("datetime")])
    pool.close()
    pool.join()
    results = [i for i in results if i is not None]
    return pd.concat(results)

dataset = multi_add_env(dataset)
dataset = dataset.astype("float")
args.env_size = len(dataset.columns) - args.feat_dim - 1

def main(args):

    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create model
    #feature selection module
    feature_reconstructor = FeatureReconstructor(feat_dim=args.feat_dim)
    feature_mask = FeatureMask(feat_dim=args.feat_dim, hidden_dim=args.feat_dim)
    
    #environment-agnostic module
    feature_extractor = FeatureExtractor(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim)
    factor_encoder = FactorEncoder(factor_dims=args.factor_dim, num_portfolio=args.feat_dim, hidden_dim=args.hidden_dim)
    alpha_layer = AlphaLayer(args.hidden_dim)
    beta_layer = BetaLayer(args.hidden_dim, args.factor_dim)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_prior_model = FatorPrior(args.batch_size, args.hidden_dim, args.factor_dim)
    predictor = Predictor(feature_extractor, factor_encoder, factor_decoder, factor_prior_model,args)

    #environment-awareness module
    env_feature_extractor = FeatureExtractor(feat_dim=args.feat_dim+args.env_size, hidden_dim=args.hidden_dim)
    env_factor_encoder = FactorEncoder(factor_dims=args.factor_dim, num_portfolio=args.feat_dim, hidden_dim=args.hidden_dim)
    env_alpha_layer = AlphaLayer(args.hidden_dim)
    env_beta_layer = BetaLayer(args.hidden_dim, args.factor_dim)
    env_factor_decoder = FactorDecoder(env_alpha_layer, env_beta_layer)
    env_factor_prior_model = FatorPrior(args.batch_size, args.hidden_dim, args.factor_dim)
    env_predictor = Predictor(env_feature_extractor, env_factor_encoder, env_factor_decoder, env_factor_prior_model,args)

    # create dataloaders
    # Dynamic batch size 
    train_ds = StockDataset(dataset,train_index)
    valid_ds = StockDataset(dataset,valid_index)
    test_ds = StockDataset(dataset,valid_index)

    train_batch_sizes = pd.DataFrame([ i[0] for i in dataset.index[train_index[:,0]].values]).value_counts(sort=False).values
    train_batch_sampler = DynamicBatchSampler(train_ds,train_batch_sizes)
 
    valid_batch_sizes = pd.DataFrame([ i[0] for i in dataset.index[valid_index[:,0]].values]).value_counts(sort=False).values
    valid_batch_sampler = DynamicBatchSampler(valid_ds,valid_batch_sizes)
    
    test_batch_sizes = pd.DataFrame([ i[0] for i in dataset.index[test_index[:,0]].values]).value_counts(sort=False).values
    test_batch_sampler = DynamicBatchSampler(test_ds,test_batch_sizes)

    train_dataloader = DataLoader(train_ds, batch_sampler=train_batch_sampler, shuffle=False, num_workers=4)
    valid_dataloader = DataLoader(valid_ds, batch_sampler=valid_batch_sampler, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_ds, batch_sampler=test_batch_sampler, shuffle=False, num_workers=4)
    device = args.device

    predictor.to(device)
    best_rankic = 0

    featrue_optimizer = torch.optim.Adam(list(feature_reconstructor.parameters())+list(feature_mask.parameters()), lr=args.lr)
    featrue_scheduler = torch.optim.lr_scheduler.OneCycleLR(featrue_optimizer,pct_start=0.1, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs//3)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,pct_start=0.1, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs//3)

    env_optimizer = torch.optim.Adam(env_predictor.parameters(), lr=args.lr)
    env_scheduler = torch.optim.lr_scheduler.OneCycleLR(env_optimizer, max_lr=args.lr,pct_start=0.1, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs//3)


    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss,pred_loss,env_loss,env_pred_loss,diff_loss,self_pred_loss,recon_diff_loss,kl_diff_loss,rank_loss,env_rank_loss,kl_loss,env_kl_loss,rank_diff_loss = train(feature_reconstructor,feature_mask,predictor,env_predictor, train_dataloader, featrue_optimizer, optimizer,env_optimizer,featrue_scheduler,scheduler,env_scheduler,args,epoch=epoch)
        val_loss,val_pred_loss,val_rank_loss,val_kl_loss,avg_rankic = validate(feature_mask,predictor, valid_dataloader,args)
        path = path = epoch % 3
        print(f"Epoch {epoch+1}: ",{"Validation Toal Loss": round(val_loss,6),"Validation Pred Loss": round(val_pred_loss,6),"Validation Ranking Loss": round(val_rank_loss,6),"Validation KL Loss": round(val_kl_loss,6),"Validation RankIC":round(avg_rankic,6)})
        if args.wandb:
            wandb.log({"Validation Toal Loss": val_loss,"Validation Pred Loss": val_pred_loss,"Validation Ranking Loss": val_rank_loss,"Validation KL Loss": val_kl_loss,"Validation RankIC":avg_rankic}, step=epoch)
            if path == 0:
                wandb.log({"Different Loss": diff_loss,"Self Reconstruction Loss":self_pred_loss,"Reconstruction Diff Loss":recon_diff_loss,"KL Diff Loss":kl_diff_loss}, step=epoch)
            elif path in [1]:
                wandb.log({"No Env Loss": train_loss,"No Env Pred Loss": pred_loss,"No Env Ranking Loss": rank_loss,"No Env KL Loss": kl_loss}, step=epoch)
            elif path in [2]:
                wandb.log({"With Env Loss": env_loss,"With Env Pred Loss": env_pred_loss,"With Env Ranking Loss": env_rank_loss,"With Env KL Loss": env_kl_loss}, step=epoch)

        if path == 0:
            print(f"Epoch {epoch+1}: ",{"Different Loss": round(diff_loss,6),"Self Reconstruction Loss":round(self_pred_loss,6),"Reconstruction Diff Loss":round(recon_diff_loss,6),"KL Diff Loss":round(kl_diff_loss,6),"Rank Diff Loss":round(rank_diff_loss,6)})
        elif path in [1]:
            print(f"Epoch {epoch+1}: ",{"No Env Loss": round(train_loss,6),"No Env Pred Loss": round(pred_loss,6),"No Env Ranking Loss": round(rank_loss,6),"No Env KL Loss": round(kl_loss,6)})
        elif path in [2]:
            print(f"Epoch {epoch+1}: ",{"With Env Loss": round(env_loss,6),"With Env Pred Loss": round(env_pred_loss,6),"With Env Ranking Loss": round(env_rank_loss,6),"With Env KL Loss": round(env_kl_loss,6)})

        if avg_rankic > best_rankic:
            best_rankic = avg_rankic
            predictor_root = os.path.join(args.save_dir, f'best_predictor_{args.run_name}_{epoch}.pt')
            feat_mask_root = os.path.join(args.save_dir, f'best_feat_mask_{args.run_name}_{epoch}.pt')
            torch.save(predictor.state_dict(), predictor_root)
            torch.save(feature_mask.state_dict(), feat_mask_root)
    
    #loading the best model for the final test set
    predictor.load_state_dict(torch.load(predictor_root))
    feature_mask.load_state_dict(torch.load(feat_mask_root))
    output = generate_prediction_scores(feature_mask,predictor,test_dataloader,args)
    output.index = dataset.index[test_index[:,-1]]
    output["label"] = dataset.loc[output.index,'label']
    print("Test Result:")
    rankic(output)
    
    if args.wandb:
        wandb.log({"Best Validation RankIC": best_rankic})
        wandb.finish()

if __name__ == '__main__':
    main(args)
