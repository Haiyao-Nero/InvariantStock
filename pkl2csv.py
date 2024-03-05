import pickle as pkl
import pandas as pd
import numpy as np

# Unpack the training dataset
with open("data\\train_all_QLIB_False_NORM_False_CHAR_False_LEN_20.pkl", "rb") as f:
    object = pkl.load(f)
    
# Print the training dataset 
df = pd.DataFrame(object)
print(df)

# Export the training dataset to CSV
df.to_csv("unpacked.csv")