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

def mlp0(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=None)
        x = tf.nn.leaky_relu(x, alpha=0.1)
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

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    h = -tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1)  # exact entropy
    return pi, logp, logp_pi, h


# def mlp_gaussian_policy0(x, a, hidden_sizes, activation, output_activation, action_space):
#     act_dim = a.shape.as_list()[-1]
#     mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
#     log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
#     std = tf.exp(log_std)
#     pi = mu + tf.random_normal(tf.shape(mu)) * std
#     logp = gaussian_likelihood(a, mu, log_std)
#     logp_pi = gaussian_likelihood(pi, mu, log_std)
#     h = tf.reduce_sum(tf.ones(shape=tf.shape(mu))*(np.log(2*np.pi*np.e)+2*log_std)/2,axis=1)  # exact entropy
#     return pi, logp, logp_pi, h
#
#
# LOG_STD_MAX = 2
# LOG_STD_MIN = -20
#
# def mlp_gaussian_policy0(x, a, hidden_sizes, activation, output_activation, action_space):
#
#     act_dim = a.shape.as_list()[-1]
#     net = mlp(x, list(hidden_sizes), activation, activation)
#     mu = tf.layers.dense(net, act_dim, activation=output_activation)
#
#     log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
#     log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
#
#     std = tf.exp(log_std)
#     pi = mu + tf.random_normal(tf.shape(mu)) * std
#     logp = gaussian_likelihood(a, mu, log_std)
#     logp_pi = gaussian_likelihood(pi, mu, log_std)
#     h = tf.reduce_sum(tf.ones(shape=tf.shape(mu)) * (np.log(2 * np.pi * np.e) + 2 * log_std) / 2, axis=1)  # exact entropy
#     return pi, logp, logp_pi, h
#
# LOG_STD_DELTA = 1.0
# def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
#
#     act_dim = a.shape.as_list()[-1]
#     net = mlp(x, list(hidden_sizes), activation, activation)
#     mu = tf.layers.dense(net, act_dim, activation=output_activation)
#
#     log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(act_dim, dtype=np.float32))
#
#     # scheme1
#     # log_std1 = tf.layers.dense(net, act_dim, activation=tf.tanh) * LOG_STD_DELTA
#     # scheme2
#     log_std1 = tf.layers.dense(net, act_dim, activation=tf.tanh)
#     log_std1 = -LOG_STD_DELTA * 0.5*(1-log_std1)
#
#     log_std = log_std + log_std1
#
#     std = tf.exp(log_std)
#     pi = mu + tf.random_normal(tf.shape(mu)) * std
#     logp = gaussian_likelihood(a, mu, log_std)
#     logp_pi = gaussian_likelihood(pi, mu, log_std)
#     h = tf.reduce_sum(tf.ones(shape=tf.shape(mu)) * (np.log(2 * np.pi * np.e) + 2 * log_std) / 2, axis=1)  # exact entropy
#     return pi, logp, logp_pi, h


def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, act_dim, hidden_sizes, activation, output_activation):

    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi, log_std

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    act_dim = a.shape.as_list()[-1]

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        # pi, logp, logp_pi, h = policy(x, a, hidden_sizes, activation, output_activation, action_space)
        mu, pi, logp_pi, log_std = policy(x, act_dim, hidden_sizes, activation, output_activation)

        a_mu = 0.5 * tf.log((1 + a / action_scale) / (1 - a / action_scale + 1e-10))  # tanh-1
        logp_a = gaussian_likelihood(a_mu, mu, log_std)
        _, _, logp_a = apply_squashing_func(mu, a_mu, logp_a)

        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    mu *= action_scale
    pi *= action_scale

    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp_a, logp_pi, -logp_pi, v