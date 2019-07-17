import gym
import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete, Tuple
import cv2



class Env_wrapper(gym.Env):

    def __init__(self, env, flag="obs", action_repeat=1):
        self.env = gym.make(env)
        self.action_repeat = action_repeat
        self.flag = flag
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        if self.flag == "obs_act":
            # print("<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            self.act_dim = self.action_space.shape[0]
            self.obs_dim = self.action_space.shape[0] + self.env.observation_space.shape[0]
            self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        if self.flag == "obs_act":
            obs = np.append(obs, np.zeros(self.act_dim))
        return obs

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            obs, r, done, info = self.env.step(action)
            # r -= 0.001  # punishment for stay still
            reward += r
        # reward -= 0.001
        if self.flag == "obs_act":
            obs = np.append(obs, action.reshape(self.act_dim))
        reward = np.clip(reward, -3, 1000)
        return obs, reward, done, info

    def render(self):
        self.env.render()


if __name__ == "__main__":
    for _ in range(4):
        env = Env_wrapper("BipedalWalkerHardcore-v2")
        obs = env.reset()

        done = False
        i = 0
        total_reward = 0.0
        while not done:
            i += 1
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
            print(reward)
            # print(obs.shape, env.obs_dim)
        env.close()
