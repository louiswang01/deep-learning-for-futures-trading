
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim

import random
import timeit
import os.path
import pandas as pd
import numpy as np
import datetime as dt
from collections import deque

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
         cash_idx = None): # cash is last item of input
    self.num_actions = num_actions
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



def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
  episode_rewards = []

  for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
      action = agent.get_action(state)
      next_state, reward, done, _ = env.step(action)
      agent.replay_buffer.push(state, action, reward, next_state, done)
      episode_reward += reward

      if len(agent.replay_buffer) > batch_size:
        agent.update(batch_size)   

      if done or step == max_steps-1:
        episode_rewards.append(episode_reward)
        print('Episode ' + str(episode) + ': ' + str(episode_reward))
        break

      state = next_state

  return episode_rewards



class BasicBuffer:
  def __init__(self, max_size):
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
    experience = (state, action, np.array([reward]), next_state, done)
    # experience = (state, action, reward, next_state, done)
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



# Ornstein-Ulhenbeck Noise
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
  def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
    self.mu           = mu
    self.theta        = theta
    self.sigma        = max_sigma
    self.max_sigma    = max_sigma
    self.min_sigma    = min_sigma
    self.decay_period = decay_period
    self.action_dim   = action_space.shape[0]
    self.low          = action_space.low
    self.high         = action_space.high
    self.reset()
      
  def reset(self):
    self.state = np.ones(self.action_dim) * self.mu
      
  def evolve_state(self):
    x  = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
    self.state = x + dx
    return self.state
  
  def get_action(self, action, t=0):
    ou_state = self.evolve_state()
    self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
    return np.clip(action + ou_state, self.low, self.high)

# input (state, action(in our case action is a float))
# output: float
class Critic(nn.Module):
  def __init__(self, obs_dim, action_dim):
    super(Critic, self).__init__()

    self.obs_dim = obs_dim
    self.action_dim = action_dim

    self.linear1 = nn.Linear(self.obs_dim, 40)
    self.linear2 = nn.Linear(40 + self.action_dim, 30)
    self.linear3 = nn.Linear(30, 10)
    self.linear4 = nn.Linear(10, 1)

  def forward(self, x, a):
    x = F.relu(self.linear1(x))
    xa_cat = torch.cat([x,a], 1)
    xa = F.relu(self.linear2(xa_cat))
    xa = F.relu(self.linear3(xa))
    qval = self.linear4(xa)

    return qval

# input: state
# output: action (in our case is a float)
class Actor(nn.Module):

  def __init__(self, obs_dim, action_dim):
    super(Actor, self).__init__()

    self.obs_dim = obs_dim
    self.action_dim = action_dim

    self.linear1 = nn.Linear(self.obs_dim, 40)
    self.linear2 = nn.Linear(40, 30)
    self.linear3 = nn.Linear(30, self.action_dim)

  def forward(self, obs):
    x = F.relu(self.linear1(obs))
    x = F.relu(self.linear2(x))
    x = torch.tanh(self.linear3(x))

    return x

class DDPGAgent:
    
  def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.env = env
    # self.obs_dim = env.observation_space.shape[0] # 3, dim for NN input
    # self.action_dim = env.action_space.shape[0]  # 1, dim for NN output (action value in continuous space)
    self.obs_dim = env.get_state_shape() # input to NN
    self.action_dim = env.get_num_actions() # 1 (>0 buy, 0 hold, <0 sell)
    self.input_dimension = env.get_input_shape() # full input with ref data

    # hyperparameters
    self.env = env
    self.gamma = gamma
    self.tau = tau
    
    # initialize actor and critic networks
    self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
    self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
    
    self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
    self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)

    # Copy critic target parameters
    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      target_param.data.copy_(param.data)
    
    # optimizers
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

    self.replay_buffer = BasicBuffer(buffer_maxlen)        
    # self.noise = OUNoise(self.env.action_space)
      
  def get_action(self, obs):
    state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
    action = self.actor.forward(state)
    action = action.squeeze(0).cpu().detach().numpy()
    return action
  
  def update(self, batch_size):
    states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
    state_batch = torch.FloatTensor(state_batch).to(self.device)
    action_batch = torch.FloatTensor(action_batch).to(self.device)
    reward_batch = torch.FloatTensor(reward_batch).to(self.device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
    masks = torch.FloatTensor(masks).to(self.device)

    curr_Q = self.critic.forward(state_batch, action_batch)
    next_actions = self.actor_target.forward(next_state_batch)
    next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
    expected_Q = reward_batch + self.gamma * next_Q
    
    # update critic
    q_loss = F.mse_loss(curr_Q, expected_Q.detach())

    self.critic_optimizer.zero_grad()
    q_loss.backward() 
    self.critic_optimizer.step()

    # update actor
    policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
    
    self.actor_optimizer.zero_grad()
    policy_loss.backward()
    self.actor_optimizer.step()

    # update target networks 
    for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
      target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
      target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


