"""
使用Double DQN进行训练
使用两个输入一样的DQN网络：DQN TARGET
DQN用来选取价值最大的动作
TARGET用来计算TD Target
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as py
import pandas as pd
import gym

# 初始化参数
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9                 # 奖励衰减系数
TARGET_REPLACE_TIER = 100   # 更新频率
MEMORY_CAPACITY = 2000      # replay大小
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTION = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTION)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        print('初始化')
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_target = 0                                   # target net更新步骤记录
        self.memory_counter = 0                                      # replay已存放的数据量
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化replay
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_fun = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 只输入一个样本
        if np.random.uniform() < EPSILON: # greedy
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy() # 选出数值最大的输出，1代表每行，0代表每列，[1]代表只需要数组小便 后面代表转换成array数组
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) # 返回action
        else:   # 随机
            action = np.random.randint(0, N_ACTION)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) # 返回action

        return action

    def store_transition(self, s, a, r, s_):
        # 存储一个transition
        transition = np.hstack((s, [a, r], s_)) # 拼接数组
        # 替换调replay中最旧的transition
        index = self.memory_counter % MEMORY_CAPACITY   # 应存储在哪个位置
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net参数更新
        if self.learn_step_target % TARGET_REPLACE_TIER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # 直接赋值更新权重
        self.learn_step_target += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # 随机抽样
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])                          # 取从index 到 N_STATES的值
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))   # 取每个index N_STATES到N_STATES+1
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # eval中w.r.t在经验中的动作
        q_eval = self.eval_net(b_s).gather(1, b_a) # gather第一个参数： 0为行，1为列 b_a作为索引
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_fun(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
# 算法更新
for epoch in range(400):
    s = env.reset()
    ep_r  = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # 修改奖励(原奖励不好)
        # 杆子越便向中间，车越便向旁边r越小 车越靠中间r越大
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:    # replay满了开始学习
            dqn.learn()
            if done:
                print('Ep: ',epoch,
                      '| Ep_r', round(ep_r, 2))

        if done:
            break

        s = s_


