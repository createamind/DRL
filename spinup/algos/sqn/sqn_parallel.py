import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
# from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D, AveragePooling2D, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm

import gym


SHOW_PREVIEW = False
# IM_WIDTH = 640
# IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = int(1e6)
MIN_REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "EXP3_Pong"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = int(1e5)

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.9975 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):

        # self.writer.add_summary(stats, self.step)
        for name in stats:
             summary = tf.Summary(value=[
                     tf.Summary.Value(tag=name, simple_value=stats[name]),])
             self.writer.add_summary(summary, self.step)
        # self._write_logs(stats, self.step)
        self.writer.close()

    # reward wrapper
class Wrapper(object):

    def __init__(self, env, action_repeat=3, norm=True):
        self._env = env
        self.action_repeat = action_repeat
        self.norm = norm

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        _ = self._env.reset()
        obs = self._env.step(1)[0].astype(float)        # auto start
        obs = (obs-128)/128 if self.norm else obs
        return obs

    def step(self, action):
        # action +=  args.act_noise * (-2 * np.random.random(4) + 1)
        r = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self._env.step(action)

            obs = obs.astype(float)
            obs = (obs-128)/128 if self.norm else obs
            # obs, reward, done, info = self._env.step(action+1)  # Discrete(3)
#                if info['ale.lives'] < 5:
#                    done = True
#                else:
#                    done = False

            r = r + reward

            if done:
                return obs, r, done, info
#                 print(obs)
        return obs, r, done, info


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def model_cnn(self, input_shape):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())

        return model.input, model.output

    def model_mlp(self, input_shape):
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(400, activation='relu', input_dim=input_shape))
        model.add(Dense(300, activation='relu'))
        # model.add(Dense(10, activation=None))

        return model

    def create_model(self):
        # base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))
        base_model = self.model_mlp(self.observation_dim)

        x = base_model.output
        # x = GlobalAveragePooling2D()(x)

        predictions = Dense(self.action_dim, activation="linear")(x)    # predict the Q value for each action
        model = Model(inputs=base_model.input, outputs=predictions)     # Q(a|s)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, self.observation_dim)).astype(np.float32)
        y = np.random.uniform(size=(1, self.action_dim)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            # time.sleep(0.01)


if __name__ == '__main__':
    FPS = 120
    # For stats
    ep_rewards = []
    ep_fps = []

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    # env = gym.make('CartPole-v0')
    # env = gym.make('Pong-ram-v0')
    env = gym.make('Pong-ram-v0')
    env = Wrapper(env, action_repeat=2, norm=True)
    agent = DQNAgent(env)

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones(env.observation_space.shape[0]))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
        episode_start_time = time.time()
        env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only

        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                episode_time = time.time() - episode_start_time
                break
                # episode_end_time = time.time()

        # End of episode - destroy agents
        ep_rewards.append(episode_reward)
        ep_fps.append(step/episode_time)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            fps = sum(ep_fps[-AGGREGATE_STATS_EVERY:])/len(ep_fps[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward,
                                           reward_min=min_reward,
                                           reward_max=max_reward,
                                           epsilon=epsilon,
                                           fps=fps)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')