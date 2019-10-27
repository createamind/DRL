import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(alpha, x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    v_x = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    pi_log = tf.nn.log_softmax(v_x / alpha, axis=1)
    mu = tf.argmax(pi_log, axis=1)
    pi = tf.squeeze(tf.random.multinomial(pi_log, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * pi_log, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * pi_log, axis=1)
    hx = tf.reduce_sum(tf.exp(pi_log)*pi_log, axis=1)                     # exact entropy
    vx = tf.reduce_sum(tf.exp(pi_log) * v_x, axis=1)                   # value
    q = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * v_x, axis=1)
    return mu, pi, hx, vx, q, logp, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(alpha, x, a, hidden_sizes=(64,64), activation=tf.nn.relu,
                     output_activation=None, policy=None, action_space=None):


    policy = mlp_categorical_policy

    with tf.variable_scope('q'):
        mu, pi, hx, vx, q, logp, logp_pi = policy(alpha, x, a, hidden_sizes, activation, output_activation, action_space)
    return mu, pi, logp, logp_pi, vx, q, hx