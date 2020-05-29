import pandas as pd
import numpy as np
import random
import os

random.seed(123)
# In the future: command line args
raw_data_path = 'pkls/mafia_raw.pkl'
data_save_dir = 'data'


df = pd.read_pickle(raw_data_path, compression="gzip")
grouped_df = df.groupby(["author", "game_id"])
groups = [df for _, df in grouped_df]
random.shuffle(groups)

train = groups[:int(len(groups) * 0.8)]
test = groups[int(len(groups) * 0.8):]

train = pd.concat(train).reset_index(drop=True)
test = pd.concat(test).reset_index(drop=True)

if not os.path.isdir(data_save_dir):
	os.makedirs(data_save_dir)

train.to_pickle(data_save_dir + '/train.pkl')
test.to_pickle(data_save_dir + '/test.pkl')