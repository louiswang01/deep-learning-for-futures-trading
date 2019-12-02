
import pandas as pd
import numpy as np
import datetime as dt
import timeit
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

IH_input_dir = '/content/drive/My Drive/DL_Project/input_df_cross_assets_v4/'



class Input_Sampler:
  def __init__(self, x_col_names=['mid_lag_01s', 'mid_lag_05s'], 
         ref_col_names=['mid', 'bid1', 'ask1', 'datetime'], 
         sample_start_str='20180101',
         sample_end_str='20181031',
         step_length = dt.timedelta(seconds=2),
         terminal_length = dt.timedelta(minutes=1),
         timestamp_idx = None,  # current timestamp, last entry of reference data
         trade_enter_time_idx = None,
         price_idx = None, # current mid, first entry of reference data
         cash_idx = None):  # cash is last item of input
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

    self.morning_start = dt.timedelta(hours=9, minutes=30)
    self.morning_end = dt.timedelta(hours=11, minutes=30)
    self.afternoon_start = dt.timedelta(hours=13)
    self.afternoon_end = dt.timedelta(hours=15)

    self.sample_morning_start = self.morning_start
    self.sample_morning_end = self.morning_end - terminal_length
    self.sample_afternoon_start = self.afternoon_start
    self.sample_afternoon_end = self.afternoon_end - terminal_length
  
  # total input (state + reference data)
  def get_input_shape(self):
    return self.input_len

  # input to NN
  def get_state_shape(self):
    return self.state_len
  
  # return a (randomly sampled new) single state - numpy array (self.state_len, )
  def sample_state(self):
    file_name = 'NaN'
    while(not os.path.exists(IH_input_dir + file_name)):
      trade_date = pd.to_datetime(np.random.choice(
          pd.date_range(self.sample_start_str, self.sample_end_str)))
      file_name = 'input_' + trade_date.strftime('%Y%m%d') + '.csv.gz' 

    df = pd.read_csv(IH_input_dir + file_name)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # use AM is 0, otherwise use PM
    if np.random.randint(0, 2) == 0: # random from {0, 1}
      df = df[(df['datetime']>trade_date+self.sample_morning_start) &
           (df['datetime']<trade_date+self.sample_morning_end)]
    else:
      df = df[(df['datetime']>trade_date+self.sample_afternoon_start) &
           (df['datetime']<trade_date+self.sample_afternoon_end)]
    
    sample_slice = df.sample(n=1)
    # timestamp_item = pd.to_datetime(sample_slice['datetime'])

    return np.array(list(sample_slice[self.x_col_names].values[0]) + # actual features
            [0] +  # position holding
            list(sample_slice[self.ref_col_names].values[0]) + # reference data
            [pd.NaT, 0.0]) # reference data: time entered pos, cash

    #return np.random.normal(0, 1, self.state_len)
  
  # return next_state, immediate return, done or not, info dict
  # to improve speed
  # Assumpotion: trade at mid
  def get_feedback(self, s, a):
    state_dt = s[self.timestamp_idx]
    trade_enter_time = s[self.trade_enter_time_idx]
    cash = s[self.cash_idx]
    state_date = state_dt.date()
    state_date_start = pd.to_datetime(state_date)

    file_name = 'input_' + state_date.strftime('%Y%m%d') + '.csv.gz' 
    df = pd.read_csv(IH_input_dir + file_name)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # s in morning session
    if state_dt < (state_date_start + self.morning_end):
      df = df[(df['datetime']>=state_dt) &
           (df['datetime']<(state_date_start + self.morning_end))]
    # s in afternoon session
    else:
      df = df[(df['datetime']>=state_dt) &
           (df['datetime']<(state_date_start + self.afternoon_end))]

    # terminal state
    if (state_dt + self.step_length >= df['datetime'].iloc[-1]) or \
      (state_dt + self.step_length >= trade_enter_time + self.terminal_length):
        df = df[df['datetime']<=(state_dt + self.step_length)]
        df = df.iloc[-1]
        r = s[self.pos_idx] * (df['mid'] - s[self.price_idx])
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx]] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash]) # reference data: time entered pos, cash
        done = True
    else:
      df = df[df['datetime']<=(state_dt + self.step_length)]
      df = df.iloc[-1]
      done = False
      r = s[self.pos_idx] * (df['mid'] - s[self.price_idx])
      # buy 1 share
      if a == 0:
        if trade_enter_time is pd.NaT:  # first time enter position
          trade_enter_time = df['datetime']
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx] + 1] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash - df['mid']]) # reference data: time entered pos, cash
      # hold 
      elif a == 1:
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx]] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash]) # reference data: time entered pos, cash
      
      # sell a == 2
      else:
        if trade_enter_time is pd.NaT:  # first time enter position
          trade_enter_time = df['datetime']
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx] - 1] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash + df['mid']]) # reference data: time entered pos, cash

    return s_, r, done, {}





# Estimates Q(s, a): input state, output Q(s, a) for each a in action space
class Net(nn.Module):
  def __init__(self, input_features_dimension, 
         num_action, hidden_size=10):
    super(Net, self).__init__()
    self.input_features_dimension = input_features_dimension
    self.num_actions = num_actions
    self.hidden_size = hidden_size

    self.fc1 = nn.Linear(self.input_features_dimension, self.hidden_size)
    self.fc1.weight.data.normal_(0, 0.1)   # initialization
    self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.fc2.weight.data.normal_(0, 0.1)   # initialization
    self.out = nn.Linear(self.hidden_size, self.num_actions)
    self.out.weight.data.normal_(0, 0.1)   # initialization

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    actions_value = self.out(x)
    return actions_value



class DQN(object):
  def __init__(self, input_features_dimension,
         input_dimension, num_actions, 
         memory_size, target_update_period, 
         batch_size, discount_gamma, hidden_size=10):

    self.eval_net = Net(input_features_dimension, num_actions, hidden_size)
    self.target_net = Net(input_features_dimension, num_actions, hidden_size)

    self.input_features_dimension = input_features_dimension
    self.input_dimension = input_dimension
    self.num_actions = num_actions
    self.memory_size = memory_size
    self.target_update_period = target_update_period
    self.batch_size = batch_size
    self.discount_gamma = discount_gamma
    
    self.learn_step_counter = 0  # for target updating
    self.memory_counter = 0    # for storing memory
    self.memory = np.zeros((self.memory_size, self.input_features_dimension * 2 + 2)) # (2000, len([s, a, r, s_next]))
    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
    self.loss_func = nn.MSELoss()

  def choose_action(self, x):
    # unsqueeze(): Returns a new tensor with a dimension of size one inserted at the specified position
    # x size: (input_dimension, ) (1d)
    # torch.unsqueeze(torch.FloatTensor(x), 0) size: torch.Size([1, input_dimension]) (2d)
    # tensor([[x_1, x_2, ..., x_input_dimension]])

    x = list(x) # torch.FloatTensor(x) won't work for np.array

    x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)) # only one sample
    # input only one sample
    if np.random.uniform() < EPSILON:   # greedy
      # actions_value is: tensor([[x_1, x_2, x_3, ..., x_input_dimension]], grad_fn=<AddmmBackward>)
      actions_value = self.eval_net.forward(x)
      # torch.max(actions_value, 1) returns a 2D structure (max taken across axis=1)
      # 1st is array of max values (each element is max value across column for a row)
      # 2nd is array of indices of max value (column index of the max col value for a row)
      # torch.max(actions_value, 1)[1] extracts the max indices
      # torch.max(actions_value, 1)[1].data.numpy() transforms it into numpy array
      # action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
      action = torch.max(actions_value, 1)[1].data.numpy()[0]
      # action = torch.max(actions_value, 1)[1].data.numpy()
      # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
    else:   # random
      action = np.random.randint(0, self.num_actions)
    return action

  def store_transition(self, s, a, r, s_):
    transition = np.hstack((s, [a, r], s_))
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition
    self.memory_counter += 1

  def learn(self):
    # target parameter update
    if self.learn_step_counter % self.target_update_period == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict())
    self.learn_step_counter += 1

    # sample batch transitions
    # extract data from memory
    # (32,) vector, choose 32 (==BATCH_SIZE) random indicies in memory
    # the random sample is generated from np.arange(self.memory_size)
    sample_index = np.random.choice(self.memory_size, self.batch_size)
    b_memory = self.memory[sample_index, :] # (32, self.input_features_dimension *2 +2)
    # torch.Size([32, self.input_features_dimension]):
    b_s = Variable(torch.FloatTensor(b_memory[:, :self.input_features_dimension]))
    # torch.Size([32, 1]):
    b_a = Variable(torch.LongTensor(b_memory[:, self.input_features_dimension:self.input_features_dimension+1].astype(int)))
    b_r = Variable(torch.FloatTensor(b_memory[:, self.input_features_dimension+1:self.input_features_dimension+2]))
    b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.input_features_dimension:]))

    # q_eval w.r.t the action in experience
    # according to action taken b_a, choose q_eval (q_eval has value for all actions)
    q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
    # self.eval_net(b_s) shape: torch.Size([32, self.num_actions]) (2D, one row for each batch item)
    # self.eval_net(b_s).gather(1, b_a)  shape: torch.Size([32, 1]) (2D, one row for each batch, 1 column in total,
    #    choose the value for row i as the b_a[i]-th column value of eval_net(b_s)'s ith row)
    #    selects the index of b_s's axis=1 based on value of b_a
    q_next = self.target_net(b_s_).detach()     # q_next does not pass error in opposite direction
    # detach from graph, don't backpropagate
    q_target = b_r + self.discount_gamma * q_next.max(1)[0]   # shape (batch, 1)
    loss = self.loss_func(q_eval, q_target)

    # calculate, and update evel_net
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

