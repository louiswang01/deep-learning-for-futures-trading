# -*- coding: utf-8 -*-
"""main-for-present.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kdVS7wA6-_KdSOKt74S--2LVl-R_emHB
"""

""" mount drive """

import os

GOOGLE_DIR = '/content/drive/My Drive/DL_Project'
LOCAL_DIR = '/home/ziyan/Desktop/Deep Learning Project'

def colab_mount_google_drive():
    drive.mount('/content/drive', force_remount=True)
    os.chdir(GOOGLE_DIR)
    os.listdir()

def mount_local_drive():
    os.chdir(LOCAL_DIR)
    os.listdir()

try:
    from google.colab import drive
    colab_mount_google_drive()
    DIR = GOOGLE_DIR
    print('Mounted google drive')
except ModuleNotFoundError:
    mount_local_drive
    DIR = LOCAL_DIR
    print('Mounted local drive')

import pandas as pd
df1 = pd.read_csv('input_df_cross_assets_v2/input_20181101.csv.gz')
df1.info()
# df1['mid_lag_30s'].iloc[3565:3580]
# df1.head()

""" model specification """

# for dataset
product = 'input_df_cross_assets_v2'
fields = [
    'wmid_mid', 'wmid_last'
    'mid_lag_1m', 'mid_lag_5min', 'mid_lag_10m',
    'wmid_ma_05m', 'wmid_ma_10m', 
    'wmid_max_05m', 'wmid_max_10m', 
    'wmid_min_05m', 'wmid_min_10m', 
    'wmid_bidask_05m',
    'total_volume_10s', 'total_volume_10m', 
    'signed_volume_10s', 'signed_volume_10m',
    'signed_tick_10s', 'signed_tick_10m',
    'IF_mid_lag_01min', 'IF_mid_lag_05m', 'IF_mid_lag_10m',
    'IC_mid_lag_01m', 'IC_mid_lag_05m', 'IC_mid_lag_10m'
]  # column names
y_field = 'mid_30s'

series_length = 60  # number of samples
sample_interval = 30  # sample every 30 seconds
cache_limit = 300

use_cuda = True

# for train and validate dataloader
params = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 3
}

train_sd = '20180102'
train_ed = '20180930'
validate_sd = '20181001'
validate_ed = '20181031'
test_sd = '20181101'
test_ed = '20181231'

""" create index table for selected product """
import re

os.chdir(os.path.join(DIR, product))
table_file = f'{product}_train_indexes_{series_length}_{sample_interval}.csv'

if not os.path.isfile(table_file):  # skip this step if the index tables already exists (only check train here)
    files = {'train': [], 'validate': [], 'test': []}  # dict of list of dicts
    
    for fn in sorted(os.listdir()):
        # rows number is first subtracted by {series_length} at the begining
        m = re.search('input_(2018\d{4})', fn)
        if m:
            date = m.group(1)
            if date <= train_ed:
                data_type = 'train'
            elif date <= validate_ed:
                data_type = 'validate'
            else:
                data_type = 'test'
            
            df = pd.read_csv(fn)
            cutoff = series_length * sample_interval / 0.5  # drop the tailing indexes
            files[data_type].append({'file_name': fn, 'rows': int(df.shape[0] - cutoff - 1)})

    for data_type in files:
        table = pd.DataFrame(files[data_type])
        table['date'] = table['file_name'].str.split('_').str[-1].str.split('.').str[0]
        table_file = f'{product}_{data_type}_indexes_{series_length}_{sample_interval}.csv'
        table.to_csv(f'{table_file}', index=False)
        print(f'Wrote {data_type} index table to {product}/{table_file}')
else:
    print('Skipped.')

os.chdir(os.path.join(DIR))

""" Create data generators """

from torch.utils import data
from dataset import Dataset

# create a train, validation and test data loader
train_set = Dataset(train_sd, train_ed, product=product, data_type='train', x_fields=fields, y_field=y_field, 
                    series_length=series_length, sample_interval=sample_interval,
                    cache_limit=cache_limit, use_cuda=use_cuda)
train_loader = data.DataLoader(train_set, **params)

validation_set = Dataset(validate_sd, validate_ed, product=product, data_type='validate', x_fields=fields, y_field=y_field,
                         series_length=series_length, sample_interval=sample_interval,
                         cache_limit=cache_limit, use_cuda=use_cuda)
validation_loader = data.DataLoader(validation_set, **params)

test_set = Dataset(test_sd, test_ed, product=product, data_type='test', x_fields=fields, y_field=y_field,
                   series_length=series_length, sample_interval=sample_interval,
                   cache_limit=cache_limit, use_cuda=use_cuda)
# for test only
test_params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 3
}
test_loader = data.DataLoader(test_set, **test_params)

""" 
Init and train model

Main issues:
1. tensor shape for LSTM (3 dimensions with batch as the middle one)
2. model (parameter) and data precision should match by using float()
"""

import gc
import torch.optim as optim
from time import time
from models import *

if use_cuda:
    model = PrototypeModel(input_size=len(fields), num_layers=series_length).float().cuda()  # use float precision instead of double
else:
    model = PrototypeModel(input_size=len(fields), num_layers=series_length).float()

criterion = nn.SmoothL1Loss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.6)
model.train()
epoch_error = {}
out = []

for epoch in range(10):
    running_loss = 0.0
    start_time = time()
    epoch_error[epoch] = []
    batch_idx = 0
    
    for data, target in train_loader:
        batch_idx += 1
        optimizer.zero_grad()
#         print(data)
        if use_cuda:
            prediction = model(data.float().cuda())
        else:
            prediction = model(data.float())
        
        if use_cuda:
            loss = criterion(prediction, target.cuda())
        else:
            loss = criterion(prediction, target)
        
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        cost_time = time() - start_time
#         print(prediction, target)
        print('[epoch: %d, batch:  %5d] target: %6.2f | pred: %6.2f | loss: %.5f  | %.2f' %
              (epoch + 1, batch_idx + 1, target, prediction, running_loss, cost_time))
        
        out.append([float(target), float(prediction)])
        running_loss = 0.0
        start_time = time()
        
        epoch_error[epoch].append(loss)
        
        # Remove this when actually training. 
        # Used to terminate early. 
        # if batch_idx > 50000: 
        #     break
            
#     gc.collect()

"""Local Evaluation / Test"""

import numpy as np

test_mode = False

model.eval()
out = []
# model.cuda()
loader = validation_loader if not test_mode else test_loader

with torch.no_grad():
    mse = []
    batch_idx = 0
    
    for data, target in loader:
        batch_idx += 1
#         print(data, target)
        prediction = model(data.float().cuda())
        mse.append(np.square(prediction.cpu() - target))
        if test_mode:
            out.append([float(target), float(prediction)])
        
        if test_mode:
            print('[epoch: %d, batch:  %5d] target: %6.2f | pred: %6.2f' %
                  (epoch + 1, batch_idx + 1, target, prediction))
        # Used to terminate early, remove.
        if batch_idx >= 1000: 
            break

print('MSE: %.2f' % (np.array(mse).mean()))
if test_mode:
    print(out)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

with open('test result_2.csv', 'w') as f:
    for lst in out:
        f.write(','.join([str(i) for i in lst])+'\n')

""" Benchmark Model """

import torch.optim as optim
from time import time
from models import *
model = LinearRegressionModel(len(fields)).cuda()
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)
batch_idx=0
out=[]
for data, target in train_loader:
        batch_idx += 1
        optimizer.zero_grad()
#         print(data)
        if use_cuda:
            prediction = model(data[-1][0].float().cuda())
        else:
            prediction = model(data[-1][0].float())
        
        if use_cuda:
            loss = criterion(prediction, target.cuda())
        else:
            loss = criterion(prediction, target)
        
        loss.backward()
        optimizer.step()
        
        print('[epoch: %d, batch:  %5d] target: %6.2f | pred: %6.2f | loss: %.5f ' %
              (1, batch_idx + 1, target, prediction, loss))
        
        out.append([float(target), float(prediction)])
        
        # Remove this when actually training. 
        # Used to terminate early. 
        # if batch_idx > 50000: 
        #     break

""" Benchmark Model Local Evaluation / Test"""

import numpy as np

test_mode = False

model.eval()
out = []
# model.cuda()
loader = validation_loader if not test_mode else test_loader

with torch.no_grad():
    mse = []
    batch_idx = 0
    
    for data, target in loader:
        batch_idx += 1
#         print(data, target)
        prediction = model(data[-1][0].float().cuda())
        mse.append(np.square(prediction.cpu() - target))
        if test_mode:
            out.append([float(target), float(prediction)])
        
        if test_mode:
            print('[epoch: %d, batch:  %5d] target: %6.2f | pred: %6.2f' %
                  (epoch + 1, batch_idx + 1, target, prediction))
        # Used to terminate early, remove.
        # if batch_idx >= 1000: 
        #     break

print('MSE: %.2f' % (np.array(mse).mean()))
if test_mode:
    print(out)