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

# 75% train, 15% dev, 15% test
train_pivot = int(len(groups) * 0.7)
test_pivot = train_pivot + int(len(groups) * 0.15)

train = groups[:train_pivot]
dev = groups[train_pivot:test_pivot]
test = groups[test_pivot:]

train = pd.concat(train).reset_index(drop=True)
dev = pd.concat(dev).reset_index(drop=True)
test = pd.concat(test).reset_index(drop=True)

if not os.path.isdir(data_save_dir):
	os.makedirs(data_save_dir)

train.to_pickle(data_save_dir + '/train.pkl', compression='gzip')
dev.to_pickle(data_save_dir + '/dev.pkl', compression='gzip')
test.to_pickle(data_save_dir + '/test.pkl', compression='gzip')
