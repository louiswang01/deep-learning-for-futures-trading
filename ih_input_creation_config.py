# -*- coding: utf-8 -*-
"""IH_Input_Creation_config.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11tzRSlk9EqVtfVFPYvhWzmj87xZP-Mnu
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

import pandas as pd
import numpy as np
import datetime as dt
import timeit
import os.path

from google.colab import drive
drive.mount('/content/drive')

# User Input
OUTPUT_DIR = '/content/drive/My Drive/DL_Project/input_df_cross_assets_v99/'

# IH Features:
IH_FEATURES = {
    'mid_lag': {
        'mid_lag_1tick' : 1,
        'mid_lag_01s': 1*2,
        'mid_lag_05s': 5*2,
        'mid_lag_10s': 10*2,
        'mid_lag_30s': 30*2,
    },
    
    'wmid_ma': {
        'wmid_ma_30s' : 30*2,
        'wmid_ma_01m': 1*60*2,
        'wmid_ma_05m': 5*60*2,
    },

    'wmid_max': {
        'wmid_max_01m': 1*60*2,
        'wmid_max_05m': 5*60*2,
    },

    'wmid_min': {
        'wmid_min_01m': 1*60*2,
        'wmid_min_05m': 5*60*2,
    },

    'mid_vol': {
        'mid_vol_30s': 30*2,
        'mid_vol_01m': 1*60*2,
        'mid_vol_05m': 5*60*2,
    },

    'total_volume': {
        'total_volume_10s': 10*2,
        'total_volume_30s': 30*2,
    },

    'signed_volume': {
        'signed_volume_10s': 10*2,
        'signed_volume_30s': 30*2,
    },

    'signed_tick': {
        'signed_tick_10s': 10*2,
        'signed_tick_30s': 30*2,
    }
}



# IC and IF Features:
CROSS_ASSET_FEATURES = {
    'mid_lag': {
        'mid_lag_01s': 1*2,
        'mid_lag_05s': 5*2,
        'mid_lag_10s': 10*2,
        'mid_lag_30s': 30*2,
    },

    'mid_vol': {
        'mid_vol_30s': 30*2,
        'mid_vol_01m': 1*60*2,
        'mid_vol_05m': 5*60*2,
    },

    'total_volume': {
        'total_volume_05s': 5*2,
        'total_volume_30s': 30*2,
        'total_volume_01m': 60*2,
    },

}

# Y: pice change
Y_HORIZONS = {
    'mid_chg': {
        'mid_1tick' : 1,
        'mid_01s': 1*2,
        'mid_05s': 5*2,
        'mid_10s': 10*2,
    },

    'wmid_chg': {
        'wmid_1tick' : 1,
        'wmid_01s': 1*2,
        'wmid_05s': 5*2,
        'wmid_10s': 10*2,
    },
}



IH_dir = '/content/drive/My Drive/DL_Project/IH/'
IF_dir = '/content/drive/My Drive/DL_Project/IF/'
IC_dir = '/content/drive/My Drive/DL_Project/IC/'

front_month_map = {1: '1802', 2: '1803', 3: '1804', 4: '1805', 5: '1806', 6: '1807', 
           7: '1808', 8: '1809', 9: '1810', 10: '1811', 11: '1812', 12: '1901'}

Morning_Start = dt.timedelta(hours=9, minutes=30)
Morning_End = dt.timedelta(hours=11, minutes=30)
Afternoon_Start = dt.timedelta(hours=13)
Afternoon_End = dt.timedelta(hours=15)



def regularize(df, fill_last, fill_zero, reg_col = 'datetime', reg_str='0.5S'):
  df = df.set_index(reg_col)
  df = df.resample(reg_str).last()
  # ffill: propagate last valid observation forward to next valid
  df[fill_last] = df[fill_last].fillna(method='ffill')
  df[fill_zero] = df[fill_zero].fillna(0)
  return df

def add_features(df):
  
  # helper start:
  df['mid'] = 0.5*(df['bid1'] + df['ask1'])
  df['wmid'] = (df['bid1']*df['askv1'] + df['ask1']*df['bidv1']) / (df['askv1'] + df['bidv1'])
  
  df['trade_dir'] = 0  # approximation
  df.loc[((df['mid'] > df['mid'].shift(1)) | (df['last'] >= df['ask1'].shift(1)) ) & 
      (df['volume'] > 0), 'trade_dir'] = 1
  df.loc[((df['mid'] < df['mid'].shift(1)) | (df['last'] <= df['bid1'].shift(1)) ) & 
      (df['volume'] > 0), 'trade_dir'] = -1

  df['signed_volume'] = df['trade_dir'] * df['volume']
  
  # df['price_volume'] = df['volume'] * df['last']
  # df['price_volume'] = df['price_volume'].replace(to_replace=0, method='ffill')
  
  df['bid_bsize'] = df['bid1'] * df['bidv1']
  df['ask_asize'] = df['ask1'] * df['askv1']
  
  df['tick_up'] = 0
  df.loc[df['mid'] > df['mid'].shift(1), 'tick_up'] = 1
  
  df['tick_down'] = 0
  df.loc[df['mid'] < df['mid'].shift(1), 'tick_down'] = 1
  
  # helper end ---------------------------------------------------------------
  
  # [Features Set]:

  # snapshot: weighted mid - simple mid:
  df['wmid_mid'] = df['wmid'] - df['mid']

  # snapshot: weighted mid - last traded price:
  df['wmid_last'] = df['wmid'] - df['last']
  
  # cur simple mid - lagged simple mid:
  for c in IH_FEATURES['mid_lag']:
    df[c] = df['mid'] - df['mid'].shift(IH_FEATURES['mid_lag'][c])
    # df['mid_lag_1tick'] = df['mid'] - df['mid'].shift(1)
  
  # cur weighted mid - MA:
  for c in IH_FEATURES['wmid_ma']:
    df[c] = df['wmid'] - df['wmid'].rolling(IH_FEATURES['wmid_ma'][c]).mean()
    # df['wmid_ma_30s'] = df['wmid'] - df['wmid'].rolling(30*2).mean()

  # cur weighted mid - min/max:
  for c in IH_FEATURES['wmid_max']:
    df[c] = df['wmid'] - df['wmid'].rolling(IH_FEATURES['wmid_max'][c]).max()
    # df['wmid_max_01m'] = df['wmid'] - df['wmid'].rolling(1*60*2).max()

  for c in IH_FEATURES['wmid_min']:
    df[c] = df['wmid'] - df['wmid'].rolling(IH_FEATURES['wmid_min'][c]).min()
    # df['wmid_min_01m'] = df['wmid'] - df['wmid'].rolling(1*60*2).min()
  
  # simple mid vol:
  for c in IH_FEATURES['mid_vol']:
    df[c] = df['mid'].rolling(IH_FEATURES['mid_vol'][c]).std()
    # df['mid_vol_30s'] = df['mid'].rolling(30*2).std()
  
  # Volume and Tick Direction:
  for c in IH_FEATURES['total_volume']:
    df[c] = df['volume'].rolling(IH_FEATURES['total_volume'][c]).sum()
    # df['total_volume_10s'] = df['volume'].rolling(10*2).sum()

  for c in IH_FEATURES['signed_volume']:
    df[c] = df['signed_volume'].rolling(IH_FEATURES['signed_volume'][c]).sum()
    # df['signed_volume_10s'] = df['signed_volume'].rolling(10*2).sum()

  for c in IH_FEATURES['signed_tick']:
    df[c] = df['tick_up'].rolling(IH_FEATURES['signed_tick'][c]).sum() - \
          df['tick_down'].rolling(IH_FEATURES['signed_tick'][c]).sum()
    
  
  # clean up helper columns:
  del df['bid_bsize']
  del df['ask_asize']
  del df['signed_volume']
  del df['tick_up']
  del df['tick_down']

  return df



def add_features_other_assets(df, ticker):
  
  # helper start:
  df['mid'] = 0.5*(df['bid1'] + df['ask1'])
  # helper end ---------------------------------------------------------------

  # cur simple mid - lagged simple mid:
  for c in CROSS_ASSET_FEATURES['mid_lag']:
    df[c] = df['mid'] - df['mid'].shift(CROSS_ASSET_FEATURES['mid_lag'][c])
    # df['mid_lag_01s'] = df['mid'] - df['mid'].shift(1*2) 
  
  # simple mid vol:
  for c in CROSS_ASSET_FEATURES['mid_vol']:
    df[c] = df['mid'].rolling(CROSS_ASSET_FEATURES['mid_vol'][c]).std()
    # df['mid_vol_30s'] = df['mid'].rolling(30*2).std()

  # total trading volume:
  # Volume and Tick Direction:
  for c in CROSS_ASSET_FEATURES['total_volume']:
    df[c] = df['volume'].rolling(CROSS_ASSET_FEATURES['total_volume'][c]).sum()
    # df['total_volume_05s'] = df['volume'].rolling(5*2).sum()
  
  return df

def add_y(df):

  for c in Y_HORIZONS['mid_chg']:
    df[c] = df['mid'].shift(-Y_HORIZONS['mid_chg'][c]) - df['mid']  # future - current
    # df['mid_1tick'] = df['mid'].shift(-1) - df['mid']  # future - current

  for c in Y_HORIZONS['wmid_chg']:
    df[c] = df['wmid'].shift(-Y_HORIZONS['wmid_chg'][c]) - df['wmid']  # future - current
    # df['wmid_1tick'] = df['wmid'].shift(-1) - df['wmid']
  
  return df



def build_one_day_IH(df, other_assets, morning_session_start, morning_session_end,
                     afternoon_session_start, afternoon_session_end):
  df.rename(columns = {' instrument': 'instrument',
                     ' datetime': 'datetime',
                     ' last': 'last',
                     ' opi': 'opi',
                     ' turnover': 'turnover',
                     ' volume': 'volume',
                     ' bid1': 'bid1',
                     ' ask1': 'ask1',
                     ' bidv1': 'bidv1',
                     ' askv1': 'askv1'}, inplace = True)
  
  df = df[['datetime', 'last', 'opi', 'turnover', 'volume', 'bid1', 'ask1', 'bidv1', 'askv1']]
  fill_last_cols = ['last', 'opi', 'bid1', 'ask1', 'bidv1', 'askv1']
  fill_zero_cols = ['turnover', 'volume']

  df['datetime'] = pd.to_datetime(df['datetime'])

  df_am = df[(df['datetime'] >= morning_session_start) & 
             (df['datetime'] <= morning_session_end)]
  df_pm = df[(df['datetime'] >= afternoon_session_start) & 
             (df['datetime'] <= afternoon_session_end)]
  
  df_am = regularize(df_am, fill_last_cols, fill_zero_cols)
  df_pm = regularize(df_pm, fill_last_cols, fill_zero_cols)
  
  df_am = add_features(df_am)
  df_pm = add_features(df_pm)
  df_am = add_y(df_am)
  df_pm = add_y(df_pm)
  
  # expand columns
  for other_asset in other_assets:
    df_am = pd.merge(df_am, other_asset['am'], how='left', left_index=True, right_index=True)
    df_pm = pd.merge(df_pm, other_asset['pm'], how='left', left_index=True, right_index=True)
  
  # merge rows
  df = pd.concat([df_am, df_pm])
  return df



def build_one_day_other_asset(df, ticker, morning_session_start, morning_session_end,
                                  afternoon_session_start, afternoon_session_end):
  df.rename(columns = {' instrument': 'instrument',
                     ' datetime': 'datetime',
                     ' last': 'last',
                     ' opi': 'opi',
                     ' turnover': 'turnover',
                     ' volume': 'volume',
                     ' bid1': 'bid1',
                     ' ask1': 'ask1',
                     ' bidv1': 'bidv1',
                     ' askv1': 'askv1'}, inplace = True)
  
  df = df[['datetime', 'last', 'opi', 'turnover', 'volume', 'bid1', 'ask1', 'bidv1', 'askv1']]
  fill_last_cols = ['last', 'opi', 'bid1', 'ask1', 'bidv1', 'askv1']
  fill_zero_cols = ['turnover', 'volume']

  df['datetime'] = pd.to_datetime(df['datetime'])

  df_am = df[(df['datetime'] >= morning_session_start) & 
             (df['datetime'] <= morning_session_end)]
  df_pm = df[(df['datetime'] >= afternoon_session_start) & 
             (df['datetime'] <= afternoon_session_end)]
  
  df_am = regularize(df_am, fill_last_cols, fill_zero_cols)
  df_pm = regularize(df_pm, fill_last_cols, fill_zero_cols)
  
  df_am = add_features_other_assets(df_am, ticker)
  df_pm = add_features_other_assets(df_pm, ticker)
  
  df_am.columns = [ticker + '_' + c for c in df_am.columns]
  df_pm.columns = [ticker + '_' + c for c in df_pm.columns]
  
  return {'am': df_am, 'pm': df_pm}



start_time = timeit.default_timer()

for trade_date in pd.date_range('20180101', '20181231'):
  contract = 'IH' + front_month_map[trade_date.month] + '_' + trade_date.strftime('%Y%m%d')
  contract_if = 'IF' + front_month_map[trade_date.month] + '_' + trade_date.strftime('%Y%m%d')
  contract_ic = 'IC' + front_month_map[trade_date.month] + '_' + trade_date.strftime('%Y%m%d')
  
  if not os.path.exists(IH_dir + contract + '.csv'):
    # print(IH_dir + contract + '.csv', ' not found')
    continue
  
  print('Processing', trade_date.date(), ' Contract:', contract)
  
  IH = pd.read_csv(IH_dir + contract + '.csv')
  IF = pd.read_csv(IF_dir + contract_if + '.csv')
  IC = pd.read_csv(IC_dir + contract_ic + '.csv')
  
  morning_start = trade_date + Morning_Start
  morning_end = trade_date + Morning_End
  afternoon_start = trade_date + Afternoon_Start
  afternoon_end = trade_date + Afternoon_End
  
   
  IF = build_one_day_other_asset(IF, 'IF', morning_start, morning_end,
                  afternoon_start, afternoon_end)
  IC = build_one_day_other_asset(IC, 'IC', morning_start, morning_end,
                   afternoon_start, afternoon_end)
  
  
  IH = build_one_day_IH(IH, [IF, IC], morning_start, morning_end,
              afternoon_start, afternoon_end)
  
  IH_dropna = IH.dropna()
  
  # IH.to_csv('/content/drive/My Drive/DL_Project/input_df_cross_assets_v2/raw_input_' + 
  #           trade_date.strftime('%Y%m%d') + '.csv.gz', compression='gzip')
  IH_dropna.to_csv(OUTPUT_DIR + 'input_' + 
          trade_date.strftime('%Y%m%d') + '.csv.gz', compression='gzip')

print('Time took: ', timeit.default_timer() - start_time)





