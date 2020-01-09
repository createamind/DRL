import ctypes
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete, Tuple
import cv2
norm_mean = [3.47198111e-05, 2.86151453e+04, 2.86124122e+04, 2.21404764e+00,
             2.86178206e+04, 2.33077564e+00, 2.86092836e+04, 2.30678425e+00,
             2.77779819e+04, 2.35608638e+00, 2.86056451e+04, 2.33744184e+00,
             2.71919533e+04, 2.33598361e+00, 2.86020539e+04, 2.40077078e+00,
             2.66998709e+04, 2.25519061e+00, 2.85984546e+04, 2.37306437e+00,
             2.61536496e+04, 2.20581904e+00, 3.54173437e+04, 2.87011691e+04,
             2.78740000e+04, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
# norm_std = [5.89224963e-03, 1.58621615e+02, 1.58657997e+02, 2.41387585e+00,
#             1.58380644e+02, 2.74668115e+00, 1.58984002e+02, 2.44332505e+00,
#             4.84929495e+03, 2.85984909e+00, 2.31858412e+02, 2.59465221e+00,
#             6.25114365e+03, 2.89065169e+00, 2.86768150e+02, 2.68007396e+00,
#             7.18465966e+03, 2.75492721e+00, 3.32724732e+02, 2.70373843e+00,
#             8.05911901e+03, 2.67430937e+00, 1.53764801e+04, 1.73033362e+02,
#             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
norm_max = [1, 28837, 28834,    32, 28837,    58, 28830,    32, 28837,    52, 28826,    32,
            28837,    52, 28821,    30, 28829,    52, 28820,    30, 28829,    38, 61104, 28837,
            27874,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0]


class TradingEnv(gym.Env):
    # soFile = "/home/gu/Desktop/game_sample/game.so"
    # expso = ctypes.cdll.LoadLibrary(soFile)

    def __init__(self, *args):
        # Create env
        self.soFile = "/home/liujb/Projects/DRL/spinup/envs/trading/trading_lib/game.so"
        self.expso = ctypes.cdll.LoadLibrary(self.soFile)
        # set obs, action space
        self.observation_space = Box(
            -1,
            1,
            shape=(40,),
            dtype=np.float32)    # 40 dims infos
        self.action_space = Discrete(30)  # 30 dims action space
        # create ctype data structure (array)
        self.arr = ctypes.c_int * 1    # (1,) int
        self.arr_len = 100
        self.arr1 = ctypes.c_int * self.arr_len   # (100,) int
        self.ctx = None
        self.infos = self.arr1()          # infos hold len 100
        self.infos_len = self.arr()       # info len hold len 1
        self.actions = self.arr1()        # action hold len 100
        self.action_len = self.arr()      # action len hold 1
        self.now_day = None
        self.rewards = self.arr1()
        self.rewards_len = self.arr()
        self.timesteps = 0

    def reset(self):
        self.timesteps = 0
        # TODO: reset from any point
        self.ctx = self.expso.CreateContext()   # create env context
        self.expso.GetActions(self.ctx, self.actions, self.action_len) # make action array [0 0 0 0...] to [0 1 2 3 ...]
        obs = self._getinfo()
        self.now_day = obs[25]  # what day is the starting point
        return obs

    def close(self):
        self.expso.ReleaseContext(self.ctx)

    def _getinfo(self):
        self.expso.GetInfo(self.ctx, self.infos, self.infos_len)
        obs = np.array(self.infos[:self.infos_len[0]], dtype=float)
        obs[1:24] = (obs[1:24] - np.array(norm_mean[1:24], dtype=float))/np.array(norm_max[1:24], dtype=float)
        return obs

    def _getreward(self, obs):
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)
        score = self.rewards[0]
        profit = self.rewards[1]
        total_pro = self.rewards[2]
        reward = profit/1e5 - score
        target = obs[26]
        actual_target = obs[27]
        reward -= np.abs(target - actual_target)  # obs[26] target [27] actual
        # print(f"profit:{profit}, score:{score}, target{target}, actual_target{actual_target}")
        # print(self.rewards[:3])
        return reward

    def step(self, action):
        # pass a discrete action
        obs, reward, done, info = self._step(action)

        return obs, reward, done, info

    def _step(self, action):
        self.timesteps += 1
        self.expso.Action(self.ctx, self.actions[action])
        self.expso.Step(self.ctx)
        obs = self._getinfo()
        reward = self._getreward(obs)
        # done if exp end or today is end
        # if we can't reset from particular day, we shouldn't reset after done
        done = obs[0] or (obs[25] != self.now_day) or (self.timesteps > 1000)

        return obs, reward, done, {}



if __name__ == "__main__":
    # for _ in range(4):
    env = TradingEnv()
    # env = gym.make("CartPole-v1")
    obs = env.reset()
    # print(obs[0].shape)

    done = False
    i = 0
    total_reward = 0.0
    obs_list = []
    while not done:
        i += 1
        obs, reward, done, info = env.step(np.random.randint(0, 30, 1)[0])
        # obs, reward, done, info = env.step(7)
        # obs_list.append(obs)
        # print(obs)
    env.close()
    # print()
    # x_mean = np.array(obs_list).mean(axis=0)
    # x_max = np.array(obs_list).max(axis=0)
    # print(x_mean)
    # print("\n")
    # print(x_max)



# soFile = "./game.so"
# expso = ctypes.cdll.LoadLibrary(soFile)
#
# for i in range(10):
#     print("start")
#
#     arr_len = 100
#     arr1 = ctypes.c_int * arr_len
#     actions = arr1()
#
#     arr = ctypes.c_int * 1
#     action_len = arr()
#
#     ctx = expso.CreateContext()
#
#     expso.GetActions(ctx, actions, action_len)  # make action array
#
#     cnt = 0
#     tradin_day = 0
#     tag = True
#     while (tag):
#         infos = arr1()
#         infos_len = arr()
#         expso.GetInfo(ctx, infos, infos_len)  # get infos in env and pass infos to array
#         print("=========start")
#         for j in range(infos_len[0]):
#             print(infos[j])
#
#         rewards = arr1()
#         rewards_len = arr()
#         expso.GetReward(ctx, rewards, rewards_len)
#
#         # for j in range(rewards_len[0]):
#         # print(rewards[j])
#         score = rewards[0]
#         profit = rewards[1]
#         total_pro = rewards[2]
#         now_day = infos[25]
#         if (tradin_day != now_day):
#             cnt = 0
#             tradin_day = now_day;
#         done = infos[0];
#
#         if (done == 1):
#             expso.ReleaseContext(ctx)
#
#             break
#
#         cnt += 1
#         if (cnt <= 10):
#             action_index = 8
#             if (cnt % 2):
#                 action_index = 7
#             expso.Action(ctx, actions[action_index]);
#
#         expso.Step(ctx);
#     print(score)
#     print(total_pro)
