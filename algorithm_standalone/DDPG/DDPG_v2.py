
import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################

MAX_EPISODES = 8000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
DECAY = 0.99    # soft replacement
MEMORY_CAPACITY = int(1e6)
BATCH_SIZE = 100

RENDER = False
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)   # (s a r s_)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,     # dimmensions of action, state
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # actor and critic
        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ema = tf.train.ExponentialMovingAverage(decay = DECAY)     # Exponential Moving Average

        # apply(var_list) creates shadow variables for all elements of var_list, and returns an Operation that updates the moving averages.
        self.target_update = [ema.apply(a_params), ema.apply(c_params)]

        # overwrite the default custom_getter function. (ema.average() to substitute the default getter)
        # a_params[0]: <tf.Variable 'Actor/l1/kernel:0' shape=(3, 30) dtype=float32_ref> (see function _build_a)
        # ema.average(a_params[0]): <tf.Variable 'Actor/l1/kernel/ExponentialMovingAverage:0' shape=(3, 30) dtype=float32_ref>
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # target actor and critic
        self.a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)    # ema.averager(var): Returns the Variable holding the average of var.
        q_ = self._build_c(self.S_, self.a_, reuse=True, custom_getter=ema_getter)

        # train actor
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        # train critic
        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        # if shape of s is (3,5), then shape of s[np.newaxis, :] is (1,3,5).
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]  # (:,s a r s_)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # before update actor and critic network, the network and target network have the same weight.
        # (the following two outputs are the same before update and differ after update.)
        #print(self.sess.run(self.a, {self.S: bs[:1,...]})[0])
        #print(self.sess.run(self.a_, {self.S_: bs[:1,...]})[0])

        # update actor and critic network
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        # update target network
        self.sess.run(self.target_update)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # Actor network
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = not reuse
        # tf.variable_scope: if reuse = True, reuse variabeles defined in the same scope.
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')   # tf.multiply returns x*y element-wise.

    # Critic network
    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = not reuse
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            #s = tf.layers.dense(s, 100, tf.nn.relu, trainable=trainable)
            #s = tf.layers.dense(s, self.s_dim, tf.nn.relu, trainable=trainable)
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high[0]

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()

Ep_reward = []
for i in range(MAX_EPISODES):
    s = env.reset()
    s0 = s
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -a_bound, a_bound)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            if ddpg.pointer > MEMORY_CAPACITY:
                Ep_reward.append(ep_reward)
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward <= -900:
            #     print('Episode:',i, s0)
            # if ep_reward > -300:RENDER = True
            break


# plot reward change and test
plt.plot(np.arange(len(Ep_reward)), Ep_reward)
plt.xlabel('Episode'); plt.ylabel('ep_reward')
#plt.ion()
plt.show()

while True:
    s = env.reset()
    for t in range(200):
        env.render()
        s = env.step(ddpg.choose_action(s))[0]

print('Running time: ', time.time() - t1)
