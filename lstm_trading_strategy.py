# -*- coding: utf-8 -*-
"""lstm_trading_strategy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Um_TVaFZyp-HrLS-9ntLSQmaZVSXOzOd
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





IH_dir = '/content/drive/My Drive/DL_Project/LSTM_Predicted_IH/'

front_month_map = {1: '1802', 2: '1803', 3: '1804', 4: '1805', 5: '1806', 6: '1807', 
           7: '1808', 8: '1809', 9: '1810', 10: '1811', 11: '1812', 12: '1901'}

Morning_Start = dt.timedelta(hours=9, minutes=30)
Morning_End = dt.timedelta(hours=11, minutes=30)
Afternoon_Start = dt.timedelta(hours=13)
Afternoon_End = dt.timedelta(hours=15)



def add_trade_strategy(df):
  upside_thresh = 0.05
  downside_thresh = -0.05
  take_profit = 2.5 * 0.2
  stop_loss = -1.5 * 0.2
  holding_period = 30*2  # 30s
  max_pos = 3
  pos = []
  enter_time = []
  enter_price = []
  pos_holding = 0
  cash = 0.0
  action = 0
  # enter_price = np.nan
  df['pos'] = 0.0
  df['cash'] = 0.0
  df['action'] = 0

  for i in df.index:
    
    # take profit / stop loss / off load existing position
    action = 0
    offload = []
    new_pos = []
    new_enter_time = []
    new_enter_price = []
    for p in range(len(pos)):
      if (((df.loc[i, 'mid'] - enter_price[p])*pos[p] < stop_loss) or
         ((df.loc[i, 'mid'] - enter_price[p])*pos[p] > take_profit) or
         ((i - enter_time[p]) >= holding_period)):
        off_load.append(p)
        if pos[p] > 0:  # offload long position
          action -= 1
          cash += df.loc[i, 'bid']
        else:  # offload short position
          action += 1
          cash -= df.loc[i, 'ask']
      else:
        new_pos.append(pos[p])
        new_enter_time.append(enter_time[p])
        new_enter_price.append(enter_price[p])
    
    pos = new_pos       
    enter_time = new_enter_time  
    enter_price = new_enter_price
    pos_holding = len(pos)


    # Enter into new positions based on prediction:
    if pos_holding < max_pos:
      if df.loc[i, 'pred_chg'] < downside_thresh:    
        pos.append(-1)
        enter_price.append(df.loc[i, 'mid'])
        enter_time.append(i)
        cash += df.loc[i, 'bid']
        action -= 1
      elif df.loc[i, 'pred_chg'] > upside_thresh:    
        pos.append(1)
        enter_price.append(df.loc[i, 'mid'])
        enter_time.append(i)
        cash -= df.loc[i, 'ask']
        action += 1


    df.loc[i, 'pos'] = sum(pos)
    df.loc[i, 'cash'] = cash
    df.loc[i, 'action'] = action
  
  df['cum_pnl'] = df['pos'] * df['mid'] + df['cash']
  return df



def build_one_day_IH(df, morning_session_start, morning_session_end,
           afternoon_session_start, afternoon_session_end,
           pnl_dict):
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

  df_am = add_trade_strategy(df_am)
  df_pm = add_trade_strategy(df_pm)
  
  pnl_dict['am'] = df_am['cum_pnl'].iloc[-1]
  pnl_dict['pm'] = df_pm['cum_pnl'].iloc[-1]

  # merge rows
  df = pd.concat([df_am, df_pm])
  return df





start_time = timeit.default_timer()

pnl_summary = {'date': [], 'am': [], 'pm': []}

for trade_date in pd.date_range('20180101', '20181231'):
  contract = 'IH' + front_month_map[trade_date.month] + '_' + trade_date.strftime('%Y%m%d')
  
  if not os.path.exists(IH_dir + contract + '.csv'):
    continue
  
  print('Processing', trade_date.date(), ' Contract:', contract)
  
  IH = pd.read_csv(IH_dir + contract + '.csv')
  
  morning_start = trade_date + Morning_Start
  morning_end = trade_date + Morning_End
  afternoon_start = trade_date + Afternoon_Start
  afternoon_end = trade_date + Afternoon_End
  
  pnl_cur = {'am': 0.0, 'pm': 0.0}

  IH = build_one_day_IH(IH, morning_start, morning_end,
              afternoon_start, afternoon_end, pnl_cur)
  
  pnl_summary['date'].append(trade_date.date())
  pnl_summary['am'].append(pnl_cur['am'])
  pnl_summary['pm'].append(pnl_cur['pm'])

  # IH_dropna = IH.dropna()
  
  IH.to_csv('/content/drive/My Drive/DL_Project/lstm_strategy/lstm_backtest_' + 
            trade_date.strftime('%Y%m%d') + '.csv')

pnl_summary = pd.DataFrame.from_dict(pnl_summary)
pnl_summary['cum_pnl'] = pnl_summary['am'] + pnl_summary['pm']
pnl_summary.to_csv('/content/drive/My Drive/DL_Project/lstm_strategy/pnl_summary.csv')
print('Time took: ', timeit.default_timer() - start_time)





