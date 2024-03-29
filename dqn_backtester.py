# -*- coding: utf-8 -*-
"""DQN_backtester.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BwkiInA3SyyX2-JURADd7c2qK_kCIJLa
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)



from google.colab import drive
drive.mount('/content/drive')



import pandas as pd
import numpy as np
import datetime as dt
import timeit
import os.path

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append('/content/drive/My Drive/DL_Project/')
from dqn import *  # in order to ba able to pickle load the model

dqn_model = pickle.load( open( '/content/drive/My Drive/DL_Project/double_dqn_rec_20191125', 'rb' ) )



class Input_Processer:
  def __init__(self, model,
         x_col_names=['mid_lag_01s', 'mid_lag_05s'], 
         ref_col_names=['mid', 'bid1', 'ask1', 'datetime'], 
         sample_start_str='20180101',
         sample_end_str='20181031',
         step_length = dt.timedelta(seconds=2),
         terminal_length = dt.timedelta(minutes=1),
         timestamp_idx = None,  # current timestamp, last entry of reference data
         trade_enter_time_idx = None,
         price_idx = None, # current mid, first entry of reference data
         cash_idx = None):  # cash is last item of input

    self.model = model
    
    self.x_col_names = x_col_names  # actual predictive features (x)
    self.ref_col_names = ref_col_names # reference data
    self.x_col_len = len(self.x_col_names)
    self.state_len = self.x_col_len + 1 # x features, pos holding (actual state for model)
    # x features, pos holding, ref data, 
    # initial time entering position, cash
    # input to NN (only the first self.state_len elements actually used in NN fitting)
    self.input_len = self.state_len + len(self.ref_col_names) + 1 + 1 
    self.sample_start_str = sample_start_str
    self.sample_end_str = sample_end_str
    self.step_length = step_length
    self.terminal_length = terminal_length
    
    self.pos_idx = self.x_col_len

    if timestamp_idx is not None:
      self.timestamp_idx = timestamp_idx
    else:
      self.timestamp_idx = len(x_col_names) + len(ref_col_names)
    
    if trade_enter_time_idx is not None:
      self.trade_enter_time_idx = trade_enter_time_idx
    else:
      self.trade_enter_time_idx = len(x_col_names) + len(ref_col_names) + 1
    
    if price_idx is not None:
      self.price_idx = price_idx
    else:
      self.price_idx = len(x_col_names) + 1
    
    if cash_idx is not None:
      self.cash_idx = cash_idx
    else:
      self.cash_idx = self.input_len - 1

  
  # total input (state + reference data)
  def get_input_shape(self):
    return self.input_len

  # input to NN
  def get_state_shape(self):
    return self.state_len
  
  # return a (randomly sampled new) single state - numpy array (self.state_len, )
  # df['datetime'] = pd.to_datetime(df['datetime'])
  def build_state(self, sample_df, i, cur_pos):
    # [actual features] + [# position holding]
    return np.array(list(sample_df.loc[i, self.x_col_names]) + [cur_pos])  

  def take_action(self, sample_df, i, cur_pos):
    x = np.array(list(sample_df.loc[i, self.x_col_names]) + [cur_pos])
    x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)) # only one sample
    actions_value = self.model.eval_net.forward(x)
    action = torch.max(actions_value, 1)[1].data.numpy()[0]
    return action



def add_trade_strategy(df, agent):
  max_pos = 1000
  pos = 0
  cash = 0.0
  action = 0

  # enter_price = np.nan
  df['pos'] = 0.0
  df['cash'] = 0.0
  df['action'] = 0

  for i in df.index:
    nn_action = agent.take_action(df, i, pos)
    # print(nn_action)
    # print(nn_action[0])
    # print(nn_action[0].item())

    # buy one share
    if nn_action==0 and pos < max_pos:
      pos += 1
      cash -= df.loc[i, 'mid']
      action = 1
    
    # sell one share
    elif nn_action==2 and pos > -max_pos:
      pos -= 1
      cash += df.loc[i, 'mid']
      action = -1
    
    # no action
    else:
      action = 0

    df.loc[i, 'pos'] = pos
    df.loc[i, 'cash'] = cash
    df.loc[i, 'action'] = action
  
  # assume can offload positions at EOD for 0 transaction costs
  df['cum_pnl'] = df['pos'] * df['mid'] + df['cash'] 

  return df





# Hyper Parameters


BACKTEST_START = '20180101'
BACKTEST_END = '20180331'

OUTPUT_FILE_HEAD = 'dqn'

TEST_OUTPUT_DIR = '/content/drive/My Drive/DL_Project/dqn_strategy/'
TEST_INPUT_DIR = '/content/drive/My Drive/DL_Project/input_df_cross_assets_v4/'

Morning_Start = dt.timedelta(hours=9, minutes=30)
Morning_End = dt.timedelta(hours=11, minutes=30)
Afternoon_Start = dt.timedelta(hours=13)
Afternoon_End = dt.timedelta(hours=15)


x_col_names_para = ['trade_dir', 'wmid_mid', 'wmid_last', 
           'mid_lag_1tick', 'mid_lag_01s', 'mid_lag_05s', 'mid_lag_30s',
           'wmid_ma_01m', 'wmid_max_05m', 'wmid_min_05m', 'mid_vol_30s',
           'signed_volume_10s', 'signed_volume_10s', 'total_volume_30s',
           'signed_tick_30s', 'spread', 'spread_10s', 'spread_01m',
           'IF_mid_lag_01s', 'IF_mid_lag_05s', 'IF_mid_vol_30s', 
           'IF_total_volume_30s', 'IF_spread_10s',
           'IC_mid_lag_01s', 'IC_mid_lag_05s', 'IC_mid_vol_30s', 
           'IC_total_volume_30s', 'IC_spread_10s']
ref_col_names_para = ['mid', 'bid1', 'ask1', 'datetime']

sample_start_str_para = '20180101'
sample_end_str_para = '20181031'
stap_length_para = dt.timedelta(seconds=2)
terminal_length_para = dt.timedelta(minutes=1)

env = Input_Processer(dqn_model, x_col_names = x_col_names_para, 
         ref_col_names = ref_col_names_para, 
         sample_start_str = sample_start_str_para,
         sample_end_str = sample_end_str_para,
         step_length = stap_length_para,
         terminal_length = terminal_length_para)
# (state + ref data)'s dimension:
# (x features, position held, ref data, time entered, cash)
N_STATES_REF = env.get_input_shape()

N_STATES = env.get_state_shape()  # x features + pos held



pnl_summary = {'date': [], 'am': [], 'pm': []}
prev_am_pnl = 0.0
prev_pm_pnl = 0.0
total_pnl = 0.0


start_time = timeit.default_timer()

start_time_epoch = timeit.default_timer()

for trade_date in pd.date_range(BACKTEST_START, BACKTEST_END):
  
  file_name = TEST_INPUT_DIR + 'input_' + trade_date.strftime('%Y%m%d') + '.csv.gz'

  if not os.path.exists(file_name):
    # print(IH_dir + contract + '.csv', ' not found')
    continue
  
  print('Processing', trade_date.date(),
     'Prev AM PnL：', prev_am_pnl,
     'Prev PM PnL:', prev_pm_pnl,
     'Prev Day PnL:', prev_am_pnl + prev_pm_pnl,
     ' Total PnL:', total_pnl,
     ' Prev Epoch Time Took', timeit.default_timer() - start_time_epoch)

  start_time_epoch = timeit.default_timer()

  df = pd.read_csv(file_name)
  
  morning_start = trade_date + Morning_Start
  morning_end = trade_date + Morning_End
  afternoon_start = trade_date + Afternoon_Start
  afternoon_end = trade_date + Afternoon_End

  pnl_dict = {'am': 0.0, 'pm': 0.0}
  
  df['datetime'] = pd.to_datetime(df['datetime'])

  df_am = df[(df['datetime'] >= morning_start) & 
         (df['datetime'] <= morning_end)]
  df_pm = df[(df['datetime'] >= afternoon_start) & 
         (df['datetime'] <= afternoon_end)]
  
  df_am = add_trade_strategy(df_am, env)
  df_pm = add_trade_strategy(df_pm, env)
  
  pnl_dict['am'] = df_am['cum_pnl'].iloc[-1]
  pnl_dict['pm'] = df_pm['cum_pnl'].iloc[-1]

  df = pd.concat([df_am, df_pm])

  pnl_summary['date'].append(trade_date.date())
  pnl_summary['am'].append(pnl_dict['am'])
  pnl_summary['pm'].append(pnl_dict['pm'])

  prev_am_pnl = pnl_dict['am']
  prev_pm_pnl = pnl_dict['pm']
  total_pnl += (prev_am_pnl + prev_pm_pnl)

  df.to_csv(TEST_OUTPUT_DIR + OUTPUT_FILE_HEAD + '_backtest_' + 
        trade_date.strftime('%Y%m%d') + '.csv')
  

pnl_summary = pd.DataFrame.from_dict(pnl_summary)
pnl_summary['cum_pnl'] = pnl_summary['am'] + pnl_summary['pm']
pnl_summary.to_csv(TEST_OUTPUT_DIR + OUTPUT_FILE_HEAD + '_pnl_summary.csv')
print('Time took: ', timeit.default_timer() - start_time)



# Turnover Analysis:
start_time = timeit.default_timer()

turnover_summary = {'date': [], 'turnover': []}

for trade_date in pd.date_range('20180101', '20181231'):
  file_dir = '/content/drive/My Drive/DL_Project/dqn_strategy/dqn_backtest_' + \
            trade_date.strftime('%Y%m%d') + '.csv'
  
  if not os.path.exists(file_dir):
    continue
  
  IH = pd.read_csv(file_dir)
  
  tov = np.sum( np.abs(IH['action']) )
  turnover_summary['date'].append(trade_date.date())
  turnover_summary['turnover'].append(tov)
  print('Processed', trade_date.date(), 'Turnover:', tov, 
        ' Time took so far:', 
        timeit.default_timer() - start_time)

turnover_summary = pd.DataFrame.from_dict(turnover_summary)
turnover_summary.to_csv('/content/drive/My Drive/DL_Project/dqn_strategy/turnover_summary.csv')
print('Time took: ', timeit.default_timer() - start_time)

