import pandas as pd
import numpy as np
import argparse
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.utils import flatten_dict
from dataclasses import dataclass
import os

#make argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./data', help='directory to save model')
parser.add_argument('--start_time', type=str, default='1990-01-01', help='start time')
parser.add_argument('--end_time', type=str, default='2022-12-31', help='end time')
parser.add_argument('--fit_end_time', type=str, default='2017-12-31', help='fit end time')

parser.add_argument('--val_start_time', type=str, default='2018-01-01', help='val start time')
parser.add_argument('--val_end_time', type=str, default='2023-12-31', help='val end time')

parser.add_argument('--seq_len', type=int, default=20, help='sequence length')

parser.add_argument('--normalize', action='store_true', help='whether to normalize')
parser.add_argument('--select_feature', action='store_true', help='whether to select feature')
parser.add_argument('--use_qlib', action='store_true', help='whether to use qlib data')
args = parser.parse_args()


def main(args):

    dataset = pd.read_pickle("data/simpleusdataset_norm_nona.pkl")
    # columns = list(dataset.columns)

    # index = pd.read_pickle("../data/index.pkl")
    # dataset = pd.merge(dataset,index,on="datetime")
    
    train = dataset.loc[dataset.index.get_level_values("datetime")<=pd.to_datetime("20181231")]
    valid = dataset.loc[(dataset.index.get_level_values("datetime")>pd.to_datetime("20181231")) & (dataset.index.get_level_values("datetime")<=pd.to_datetime("20201231"))]
    test = dataset.loc[dataset.index.get_level_values("datetime")>pd.to_datetime("20201231")]
    


    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir) 
        
    train = add_env(train)
    valid = add_env(valid)
    test = add_env(test)
    
    train.to_pickle('./data/train_all.pkl')
    valid.to_pickle('./data/valid_all.pkl')
    test.to_pickle('./data/test_all.pkl')
    
    # train=pd.read_pickle('./data/train_all.pkl')
    # valid=pd.read_pickle('./data/valid_all.pkl')
    # test=pd.read_pickle('./data/test_all.pkl')
    
    import multiprocessing
    num_processes = 4
    chunk_size = int(len(train) / num_processes)+1
    chunks = [(range(i*chunk_size,min((i+1)*chunk_size,len(train))),train) for i in range(0, num_processes)]
    # chunks = [(stock_list[i:i + chunk_size]) for i in range(0, len(stock_list), chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)
    train_result = np.concatenate([i for i in pool.starmap(get_index, chunks) if len(i)!=0])
    pool.close()
    pool.join()
    # train_list = []
    # for sublist in result:
    #     train_list.extend(sublist)
    np.save("./data/train_index.npy",train_result)
    
    num_processes = 16
    chunk_size = int(len(valid) / num_processes)+1
    chunks = [(range(i*chunk_size,min((i+1)*chunk_size,len(valid))),valid) for i in range(0, num_processes)]
    # chunks = [(stock_list[i:i + chunk_size]) for i in range(0, len(stock_list), chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)
    valid_result = np.concatenate([i for i in pool.starmap(get_index, chunks) if len(i)!=0])
    pool.close()
    pool.join()
    np.save("./data/valid_index.npy",valid_result)
    
    import multiprocessing
    num_processes = 16
    chunk_size = int(len(test) / num_processes)+1
    chunks = [(range(i*chunk_size,min((i+1)*chunk_size,len(test))),test) for i in range(0, num_processes)]
    # chunks = [(stock_list[i:i + chunk_size]) for i in range(0, len(stock_list), chunk_size)]
    pool = multiprocessing.Pool(processes=num_processes)
    test_result = np.concatenate([i for i in pool.starmap(get_index, chunks) if len(i)!=0])
    pool.close()
    pool.join()
    # valid_list = []
    # for sublist in result:
    #     valid_list.extend(sublist)
    # valid_list
    np.save("./data/test_index.npy",test_result)
    print("Success!")
    
def get_index(data_num,df):
    # df.set_index(["datetime","instrument"],inplace=True)
    df.sort_index(inplace=True)
    sequence_length = 20
    date_list = list(df.index.get_level_values("datetime").unique())
#     indexer = list(df.index)
    index_list = []
    print(data_num)
    for index in data_num:
        # index = np.random.randint(0,len(df.loc[df.index.get_level_values(0)<=date_list[-20]]))
        date, stock = df.index[index]
        if date>date_list[-sequence_length]:
            continue
        date_seq = range(date_list.index(date),date_list.index(date)+20)
        idx_list = [(date_list[i],stock) for i in date_seq]
        if not all(i in df.index for i in idx_list):
            continue
        if idx_list == []:
            continue 
        index_list.append(df.index.get_indexer(idx_list))
    # print(index_list)
    return np.array(index_list)
    
def add_env(train_df):
    month = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",]
    year = [str(i) for i in range(1990,2025)]
    # train_df.set_index(["datetime","instrument"],inplace=True)
    # train_df.sort_index(inplace=True)
    train_df[month]=0
    train_df[year]=0
    codes = train_df.index.codes
    new_levels = [pd.to_datetime(train_df.index.levels[0].astype(str),),train_df.index.levels[1]]
    new_index = pd.MultiIndex(levels=new_levels, codes=codes, names=train_df.index.names)
    train_df.index = new_index
    dates = list(train_df.index.get_level_values("datetime").unique())
    for date in dates:
        print(date)
        train_df.loc[train_df.index.get_level_values("datetime") == date,month[date.month-1]]=1
        train_df.loc[train_df.index.get_level_values("datetime") == date,year[date.year-1990]]=1
    return train_df
if __name__ == "__main__":
    main(args)