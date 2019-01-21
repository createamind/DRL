import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))
def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if space is None:
        return tf.placeholder(dtype=tf.float32,shape=(None,))
    if isinstance(space, Box):
        return tf.placeholder(dtype=tf.float32, shape=(None,space.shape[0]))
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,1))
    raise NotImplementedError
def placeholders_from_space(*args):
    return [placeholder_from_space(dim) for dim in args]



def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def mlp_ensemble_with_prior(x, hidden_sizes=(32,), ensemble_size=10, prior_scale=1.0, activation=None, output_activation=None):

    # An ensemble of Prior nets.
    priors = []
    for _ in range(ensemble_size):
        x_proxy = x
        for h in hidden_sizes[:-1]:
            x_proxy = tf.layers.dense(x_proxy, units=h, activation=activation)
        priors.append(tf.layers.dense(x_proxy, units=hidden_sizes[-1], activation=output_activation))      # outputs: 10 x shape(?, 4)

    prior_nets = tf.stop_gradient(tf.stack(priors, axis=2))                                          # outputs: shape(?, 4, 10)

    # An ensemble of Q nets.
    qs = []
    for _ in range(ensemble_size):
        x_proxy = x
        for h in hidden_sizes[:-1]:
            x_proxy = tf.layers.dense(x_proxy, units=h, activation=activation, kernel_initializer=tf.variance_scaling_initializer(2.0))
        qs.append(tf.layers.dense(x_proxy, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=tf.variance_scaling_initializer(2.0)))

    q_nets = tf.stack(qs, axis=2)

    # An ensemble of Q models.
    q_models = q_nets + prior_scale * prior_nets

    return q_models



"""
Policies
"""

def softmax_policy(alpha, v_x, act_dim, random_index=0, ensemble_size=10):

    # # random_index = tf.random_uniform(minval=0, maxval=ensemble_size, shape=[], dtype=tf.int32)
    # tf.get_variable_scope()._name = ''
    # with tf.variable_scope('random_head', reuse=True):
    #     random_index = tf.get_variable(name='random_int', shape=[], dtype=tf.int32)

    pi_log = tf.nn.log_softmax(v_x/alpha, axis=1)                      # shape(?, 4, 10)
    random_pi_log = tf.expand_dims(pi_log[...,random_index], axis=-1)    # shape(?, 4, 1)
    mu = tf.argmax(random_pi_log, axis=1)                      # shape(?, 1)

    # tf.random.multinomial( logits, num_samples, seed=None, name=None, output_dtype=None )
    # logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized log-probabilities for all classes.
    # num_samples: 0-D. Number of independent samples to draw for each row slice.
    pi = tf.random.multinomial(random_pi_log[...,0], num_samples=1)   # shape(?, 1)

    # logp_pi = tf.reduce_sum(tf.one_hot(mu, depth=act_dim) * pi_log, axis=1)  # use max Q(s,a)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim, axis=1) * pi_log, axis=1)             # shape(?, 4, 1)*shape(?, 4, 10)---reduce_sum--> shape(?,10)
    # logp_pi = tf.reduce_sum(tf.exp(pi_log)*pi_log, axis=1)                     # exact entropy

    # mu, pi: shape(?, 1),  logp_pi: shape(?,10)
    return mu, pi, logp_pi


def softmax_policy1(alpha, v_x, act_dim, ensemble_size=10):

    # random_index = tf.random_uniform(minval=0, maxval=ensemble_size, shape=[], dtype=tf.int32)
    # tf.get_variable_scope()._name = ''
    # with tf.variable_scope('random_head', reuse=True):
    #     random_index = tf.get_variable(name='random_int', shape=[], dtype=tf.int32)

    pi_log = tf.nn.log_softmax(v_x/alpha, axis=1)                      # shape(?, 4, 10)
    mu = tf.argmax(pi_log, axis=1)                      # shape(?, 10)

    # tf.random.multinomial( logits, num_samples, seed=None, name=None, output_dtype=None )
    # logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized log-probabilities for all classes.
    # num_samples: 0-D. Number of independent samples to draw for each row slice.
    pi = tf.stack([tf.squeeze(tf.random.multinomial(pi_log[...,i], num_samples=1), axis=1) for i in range(ensemble_size)], axis=1)   # shape(?, 10)

    # logp_pi = tf.reduce_sum(tf.one_hot(mu, depth=act_dim) * pi_log, axis=1)  # use max Q(s,a)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim, axis=1) * pi_log, axis=1)             # shape(?, 4, 1)*shape(?, 4, 10)---reduce_sum--> shape(?,10)
    # logp_pi = tf.reduce_sum(tf.exp(pi_log)*pi_log, axis=1)                     # exact entropy

    # mu, pi, logp_pi: shape(?,10)
    return mu, pi, logp_pi

"""
Actor-Critics
"""

def mlp_actor_critic(x, a, alpha, hidden_sizes=(400,300), ensemble_size=10, activation=tf.nn.relu,
                     output_activation=None, policy=softmax_policy, action_space=None):

    if x.shape[1] == 128:           # for Breakout-ram-v4
        x = (x - 128.0) / 128.0     # x: shape(?,128)

    act_dim = action_space.n
    a_one_hot = tf.one_hot(a, depth=act_dim, axis=1)    # a: shape(?,1), # a_one_hot: shape(?,4,1)

    # vfs
    # vf_mlp = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, None)
    vf_mlp = lambda x: mlp_ensemble_with_prior(x, list(hidden_sizes) + [act_dim], ensemble_size=ensemble_size, activation=activation, output_activation=output_activation)


    with tf.variable_scope('q1', reuse=tf.AUTO_REUSE):                                                # vf_mlp(x) for q1
        q1 = tf.reduce_sum(vf_mlp(x)*a_one_hot, axis=1)    # shape(?, 10)

    with tf.variable_scope('q1', reuse=True):                                    # reuse vf_mlp(x) for q1_pi
        v_x= vf_mlp(x)                                     # shape(?, 4, 10)
        # policy
        mu, pi, logp_pi = policy(alpha, v_x, act_dim, ensemble_size=ensemble_size)
        pi_one_hot = tf.one_hot(pi, depth=act_dim, axis=1) # shape(?, 4, 1)

        # q1_pi = tf.reduce_sum(v_x*mu_one_hot, axis=1)   # use max Q(s,a)
        q1_pi = tf.reduce_sum(v_x * pi_one_hot, axis=1)    # shape(?, 10)

    # with tf.variable_scope('q2'):
    #     q2 = tf.reduce_sum(vf_mlp(x)*a_one_hot, axis=1)
    # with tf.variable_scope('q2', reuse=True):
    #     # q2_pi = tf.reduce_sum(vf_mlp(x)*mu_one_hot, axis=1)   # use max Q(s,a)
    #     q2_pi = tf.reduce_sum(vf_mlp(x) * pi_one_hot, axis=1)

    mu = tf.squeeze(mu, axis=1)
    pi = tf.squeeze(pi, axis=1)

    # mu, pi: shape(?,),  logp_pi, q1, q1_pi: shape(?,10)
    return mu, pi, logp_pi, q1, q1_pi



