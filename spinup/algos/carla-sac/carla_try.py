import tensorflow as tf
import carla
import numpy as np
from spinup import sac
from spinup import ppo
from spinup import sqn
import tensorflow as tf
import gym
from env import CarlaEnv, ENV_CONFIG
from scenarios import TOWN2_STRAIGHT

env_fn = gym.make('MountainCarContinuous-v0')
#
# ac_kwargs = dict(hidden_sizes=[64, 12], activation=tf.nn.tanh)
#
# rl = 0.001
#
# logger_kwargs = dict(output_dir='/home/gu/playboy/data/sac/car',
#                      exp_name='dp=%(dp)f' % {'dp': rl})
#
# sac(env_fn=env_fn, ac_kwargs=ac_kwargs,
#     steps_per_epoch=1000, epochs=15, logger_kwargs=logger_kwargs, save_freq=1)


from env import CarlaEnv, ENV_CONFIG

en = CarlaEnv()
en.init_server()

print(en.action_space)

# en = CarlaEnv
# a = en()
# a.init_server()
# # obs = a.reset()
# print(a.observation_space)