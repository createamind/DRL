#@title Imports

# pip install plotnine
# pip install tensorflow-probability
# pip install dm-sonnet
# pip install wrapt


import collections
import warnings

import numpy as np
import tensorflow as tf
import sonnet as snt

import plotnine as gg


from typing import Any, Callable, Dict, List, Tuple

# Plotnine themes
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)

# Filter meaningless pandas warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=ImportWarning)




# @title (CODE) Define the _Deep Sea_ environment

TimeStep = collections.namedtuple('TimeStep', ['observation' ,'reward', 'pcont'])


class DeepSea(object):

    def __init__(self,
                 size: int,
                 seed: int = None,
                 randomize: bool = True):

        self._size = size
        self._move_cost = 0.01 / size
        self._goal_reward = 1.

        self._column = 0
        self._row = 0

        if randomize:
            rng = np.random.RandomState(seed)
            self._action_mapping = rng.binomial(1, 0.5, size)
        else:
            self._action_mapping = np.ones(size)

        self._reset_next_step = False

    def step(self, action: int) -> TimeStep:
        if self._reset_next_step:
            return self.reset()
        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]

        # Compute the reward
        reward = 0.
        if self._column == self._size - 1 and action_right:
            reward += self._goal_reward

        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._move_cost
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size - 1)

        # Compute the observation
        self._row += 1
        if self._row == self._size:
            observation = self._get_observation(self._row - 1, self._column)
            self._reset_next_step = True
            return TimeStep(reward=reward, observation=observation, pcont=0.)
        else:
            observation = self._get_observation(self._row, self._column)
            return TimeStep(reward=reward, observation=observation, pcont=1.)

    def reset(self) -> TimeStep:
        self._reset_next_step = False
        self._column = 0
        self._row = 0
        observation = self._get_observation(self._row, self._column)

        return TimeStep(reward=None, observation=observation, pcont=1.)

    def _get_observation(self, row, column) -> np.ndarray:
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1

        return observation

    @property
    def obs_shape(self) -> Tuple[int]:
        return self.reset().observation.shape

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def optimal_return(self) -> float:
        return self._goal_reward - self._move_cost





#@title Agent hyperparameters (shared between DQN/RPF)

max_episodes = 3000 #@param {type:"integer"}
epsilon = 0.1  #@param {type:"number"} (NB - only for DQN)
discount = 0.99  #@param {type:"number"}
batch_size = 128  #@param {type:"integer"}
replay_capacity = 100000  #@param {type:"integer"}
ensemble_size = 10  #@param {type:"integer"}
hidden_sizes = (20,)  #@param {type:"raw"}
prior_scale = 10  #@param {type:"number"}

deep_sea_size=20  #@param {type:"number"}

# Make environment
env = DeepSea(size=deep_sea_size, randomize=True)


#@title (CODE) Simple circular replay buffer with uniform sampling.

class Replay(object):
    """A simple ring buffer with uniform sampling."""

    def __init__(self, capacity: int):
        self._data = None
        self._capacity = capacity
        self._num_added = 0

    def add(self, transition: Tuple[Any]) -> None:
        if self._data is None:
            self._preallocate(transition)

        for d, item in zip(self._data, transition):
            d[self._num_added % self._capacity] = item

        self._num_added += 1


    def sample(self, batch_size: int = 1) -> Tuple[np.ndarray]:
        """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
        indices = np.random.randint(self.size, size=batch_size)
        return [d[indices] for d in self._data]
    @property
    def size(self) -> int:
        return min(self._capacity, self._num_added)

    def _preallocate(self, items: Tuple[Any]) -> None:
        """Assume flat structure of items."""
        items_np = [np.asarray(x) for x in items]

        if sum([x.nbytes for x in items_np]) * self._capacity > 1e9:
            raise ValueError('This replay buffer would preallocate > 1GB of memory.')

        self._data = [np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape)
                      for x in items_np]




# @title (CODE) Ensemble Q-Network
class EnsembleQNetwork(snt.AbstractModule):

    def __init__(self, hidden_sizes: Tuple[int], num_actions: int, num_ensemble: int, **mlp_kwargs):
        super(EnsembleQNetwork, self).__init__(name='ensemble')
        with self._enter_variable_scope():
            # An ensemble of MLPs.
            self._models = [snt.nets.MLP(output_sizes=hidden_sizes + (num_actions,), **mlp_kwargs)
                            for _ in range(num_ensemble)]                         # outputs: 10 x shape(?, 2)

        self._num_ensemble = num_ensemble

    def _build(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = snt.BatchFlatten()(inputs)
        # Forward all members of the ensemble and stack the output.
        return tf.stack([model(inputs) for model in self._models], axis=1)        # outputs: shape(?, 10, 2)



#@title (CODE) ModelWithPrior, training routine

class ModelWithPrior(snt.AbstractModule):
    """Given a 'model' and a 'prior', combines them together."""

    def __init__(self,
               model_network: snt.AbstractModule,
               prior_network: snt.AbstractModule,
               prior_scale: float = 1.):
        super(ModelWithPrior, self).__init__(name='model_with_prior')

        self._prior_scale = prior_scale
        with self._enter_variable_scope():
            self._model_network = model_network
            self._prior_network = prior_network

    def _build(self, inputs: tf.Tensor):
        prior_output = tf.stop_gradient(self._prior_network(inputs))
        model_output = self._model_network(inputs)

        return model_output + self._prior_scale * prior_output

    def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
        return (super(ModelWithPrior, self).get_variables(collection)
                + self._model_network.get_variables(collection)
                + self._prior_network.get_variables(collection))




#@title Train Ensemble DQN+RPF on _Deep Sea_
import time
t1 = time.time()

tf.reset_default_graph()

# Make a 'prior' network.
prior_network = EnsembleQNetwork(hidden_sizes, env.num_actions, ensemble_size)

# Make independent online and target networks.
q_model = EnsembleQNetwork(hidden_sizes, env.num_actions, ensemble_size)
target_model = EnsembleQNetwork(hidden_sizes, env.num_actions, ensemble_size)

# Combine these with the prior in the usual way.
q_network = ModelWithPrior(q_model, prior_network, prior_scale)
target_network = ModelWithPrior(target_model, prior_network, prior_scale)

# Placeholders
o_tm1 = tf.placeholder(dtype=tf.float32, shape=(None,) + env.obs_shape)
a_tm1 = tf.placeholder(dtype=tf.int32, shape=(None,))
r_t = tf.placeholder(dtype=tf.float32, shape=(None,))
pcont_t = tf.placeholder(dtype=tf.float32, shape=(None,))
o_t = tf.placeholder(dtype=tf.float32, shape=(None,) + env.obs_shape)

# Forward the networks
q_tm1 = q_network(o_tm1)  # [batchsize:B, ensemble_size:K, action_size:A]
q_t = target_network(o_t)  # [B, K, A]

# Online -> target network copy ops.
update_op = [tf.assign(w, v) for v, w in zip(q_network.get_variables(),
                                            target_network.get_variables())]

# Loss/optimization ops
one_hot_actions = tf.one_hot(a_tm1, depth=env.num_actions, axis=-1)  # [B, A]
q_value = tf.einsum('bka,ba->bk', q_tm1, one_hot_actions)  # [B, K]
q_target = tf.reduce_max(q_t, axis=-1)  # [B, K]
target = tf.expand_dims(r_t, 1) + discount * tf.expand_dims(pcont_t, 1) * q_target  # [B, K]
td_error = q_value - target
loss = tf.square(td_error)
optimizer = tf.train.AdamOptimizer()

sgd_op = optimizer.minimize(loss)
sgd_op_i  = [optimizer.minimize(loss[:,i]) for i in range(ensemble_size)]  #################


replay = Replay(capacity=replay_capacity)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    q_values = sess.make_callable(q_tm1, [o_tm1])
    q_values_i = [sess.make_callable(q_tm1[:,i,:], [o_tm1]) for i in range(ensemble_size)]     #################

    target_update = sess.make_callable(update_op, [])

    sgd = sess.make_callable(sgd_op, [o_tm1, a_tm1, r_t, pcont_t, o_t])
    sgd_i = [sess.make_callable(sgd_op_i[i], [o_tm1, a_tm1, r_t, pcont_t, o_t]) for i in range(ensemble_size)]  #################

    total_return = 0
    optimal_return = 0
    results = []
    for episode in range(100*max_episodes):

        # t1 = time.time()
        timestep = env.reset()
        observation = timestep.observation
        episode_return = 0
        active_head = np.random.randint(ensemble_size)

        # t3 = time.time()
        while timestep.pcont:
            obs = np.expand_dims(timestep.observation, 0)

            # ensemble_qs = q_values(obs)  # [B, K, A]
            # qs = ensemble_qs[:, active_head, :]
            qs = q_values_i[active_head](obs)

            action = np.argmax(qs, axis=1).squeeze()

            timestep = env.step(action)
            episode_return += timestep.reward
            new_observation = timestep.observation
            transition = (observation, action, timestep.reward,
                        timestep.pcont, new_observation)
            replay.add(transition)
            observation = new_observation

        # t4 = time.time()
        # print(t4-t3)

        # Do SGD at end of episode
        batch = replay.sample(batch_size)

        # sgd(*batch)                       # about 2000 episodes to solve the problem.
        # sgd_i[active_head](*batch)          # about 13000 episodes to solve the problem.

        # [sgd_i[i](*(replay.sample(batch_size))) for i in range(ensemble_size)]  # about 1000 episodes to solve the problem
        for i in range(ensemble_size):
            batch = replay.sample(batch_size)
            sgd_i[i](*batch)


        target_update()

        total_return += episode_return
        optimal_return += 0.99
        regret = optimal_return - total_return
        results.append({'episode': episode, 'regret': regret, 'algorithm': 'rpf'})
        if episode % 100 == 0 or episode == max_episodes - 1:
            print('Episode: {}.\tTotal return: {:.2f}.\tRegret: {:.2f}.'.format(episode, total_return, regret))

        # t3 = time.time()
        # print(t2-t1, t3-t2)



t2 = time.time()
print(t2-t1)


