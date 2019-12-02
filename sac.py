import timeit
import os.path
import pandas as pd
import numpy as np
import datetime as dt
import random
from collections import deque, namedtuple
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform

import torch.optim as optim



IH_input_dir = '/content/drive/My Drive/DL_Project/input_df_cross_assets_v4/'



class Input_Sampler:
  def __init__(self, x_col_names=['mid_lag_01s', 'mid_lag_05s'], 
         ref_col_names=['mid', 'bid1', 'ask1', 'datetime'], 
         sample_start_str='20180101',
         sample_end_str='20181031',
         step_length = dt.timedelta(seconds=2),
         terminal_length = dt.timedelta(minutes=1),
         num_actions = 1,
         timestamp_idx = None,  # current timestamp, last entry of reference data
         trade_enter_time_idx = None,
         price_idx = None, # current mid, first entry of reference data
         action_low = -2,  # lower bound of action value
         action_high = 2,  # upper bound of action value
         cash_idx = None):  # cash is last item of input
    self.num_actions = num_actions
    self.action_space_shape = (num_actions,)
    self.action_space_low = np.array([action_low], dtype='float32')
    self.action_space_high = np.array([action_high], dtype='float32')

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
  
  def get_num_actions(self):
    return self.num_actions
  
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
      if a > 0.0:
        if trade_enter_time is pd.NaT:  # first time enter position
          trade_enter_time = df['datetime']
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx] + a] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash - df['mid']]) # reference data: time entered pos, cash
      # hold 
      elif a == 0.0:
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx]] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash]) # reference data: time entered pos, cash
      
      # sell a < 0.0
      else:
        if trade_enter_time is pd.NaT:  # first time enter position
          trade_enter_time = df['datetime']
        s_ = np.array(list(df[self.x_col_names].values) + # actual features
            [s[self.pos_idx] + a] +  # position holding
            list(df[self.ref_col_names].values) + # reference data
            [trade_enter_time, cash + df['mid']]) # reference data: time entered pos, cash

    return s_, r, done, {}





# Buffer
Experience = namedtuple('Experience', field_names=['state', 'policy_outputs', 'action', 'reward', 'last_state'])

class Buffer:
  def __init__(self, max_size):
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
    experience = (state, action, np.array([reward]), next_state, done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    batch = random.sample(self.buffer, batch_size)

    for experience in batch:
      state, action, reward, next_state, done = experience
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
    return len(self.buffer)



# Models
class ValueNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, init_w=3e-3):
    super(ValueNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, 40)
    self.fc2 = nn.Linear(40, 30)
    self.fc3 = nn.Linear(30, output_dim)

    self.fc3.weight.data.uniform_(-init_w, init_w)
    self.fc3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x



class SoftQNetwork(nn.Module):

  def __init__(self, num_inputs, num_actions, hidden_size=30, init_w=3e-3):
    super(SoftQNetwork, self).__init__()
    self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, 1)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state, action):
    x = torch.cat([state, action], 1)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x



class GaussianPolicy(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size=30, init_w=3e-3, log_std_min=-20, log_std_max=2):
    super(GaussianPolicy, self).__init__()
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

    self.linear1 = nn.Linear(num_inputs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)

    self.mean_linear = nn.Linear(hidden_size, num_actions)
    self.mean_linear.weight.data.uniform_(-init_w, init_w)
    self.mean_linear.bias.data.uniform_(-init_w, init_w)

    self.log_std_linear = nn.Linear(hidden_size, num_actions)
    self.log_std_linear.weight.data.uniform_(-init_w, init_w)
    self.log_std_linear.bias.data.uniform_(-init_w, init_w)

  def forward(self, state):
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))

    mean  = self.mean_linear(x)
    log_std = self.log_std_linear(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

    return mean, log_std

  def sample(self, state, epsilon=1e-6):
    mean, log_std = self.forward(state)
    std = log_std.exp()
    normal = Normal(mean, std)
    z = normal.rsample()
    log_pi = (normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)).sum(1, keepdim=True)

    return mean, std, z, log_pi





class SACAgent:

  def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.env = env
    # self.action_range = [env.action_space.low, env.action_space.high]
    # self.obs_dim = env.observation_space.shape[0]
    # self.action_dim = env.action_space.shape[0]

    self.action_range = [env.action_space_low, env.action_space_high]
    self.obs_dim = env.get_state_shape() # input to NN
    self.action_dim = env.get_num_actions() # 1 (>0 buy, 0 hold, <0 sell)
    self.input_dimension = env.get_input_shape() # full input with ref data

    # hyperparameters
    self.gamma = gamma
    self.tau = tau

    # initialize networks
    self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
    self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
    self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
    self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
    self.policy_net = GaussianPolicy(self.obs_dim, self.action_dim).to(self.device)

    # copy params to target param
    for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
      target_param.data.copy_(param)

    for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
      target_param.data.copy_(param)

    # initialize optimizers
    self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
    self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
    self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    # entropy temperature
    self.alpha = alpha
    # self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
    self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space_shape).to(self.device)).item()
    self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
    self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

    self.replay_buffer = Buffer(buffer_maxlen)


  def get_action(self, state):
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    mean, log_std = self.policy_net.forward(state)
    std = log_std.exp()

    normal = Normal(mean, std)
    z = normal.sample()
    action = torch.tanh(z)
    action = action.cpu().detach().squeeze(0).numpy()

    return action


  def update(self, batch_size):
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states).to(self.device)
    actions = torch.FloatTensor(actions).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.FloatTensor(next_states).to(self.device)
    dones = torch.FloatTensor(dones).to(self.device)
    dones = dones.view(dones.size(0), -1)

    _, _, next_zs, next_log_pi = self.policy_net.sample(next_states)
    next_actions = torch.tanh(next_zs)
    next_q1 = self.target_q_net1(next_states, next_actions)
    next_q2 = self.target_q_net2(next_states, next_actions)
    next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
    expected_q = rewards + (1 - dones) * self.gamma * next_q_target

    # q loss
    curr_q1 = self.q_net1.forward(states, actions)
    curr_q2 = self.q_net2.forward(states, actions)
    q1_loss = F.mse_loss(curr_q1, expected_q.detach())
    q2_loss = F.mse_loss(curr_q2, expected_q.detach())

    # update q networks
    self.q1_optimizer.zero_grad()
    q1_loss.backward()
    self.q1_optimizer.step()

    self.q2_optimizer.zero_grad()
    q2_loss.backward()
    self.q2_optimizer.step()

    # delayed update for policy network and target q networks
    _, _, new_zs, log_pi = self.policy_net.sample(states)
    new_actions = torch.tanh(new_zs)
    min_q = torch.min(
        self.q_net1.forward(states, new_actions),
        self.q_net2.forward(states, new_actions)
    )
    policy_loss = (self.alpha * log_pi - min_q).mean()
    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    # target networks
    for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
      target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
      target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    # update temperature
    alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.exp()



