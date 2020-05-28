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



def mlp(x, hidden_sizes=(32,), activation=None, output_activation=None):
    for h in hidden_sizes[:-1]:
        # x = tf.layers.dense(x, units=h, activation=activation)
        x = tf.layers.dense(x, units=h, activation=activation, kernel_initializer=tf.variance_scaling_initializer(2.0))#, activity_regularizer=None)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=tf.variance_scaling_initializer(2.0))#, activity_regularizer=None)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])




"""
Policies
"""

def softmax_policy(alpha, v_x, act_dim):

    pi_log = tf.nn.log_softmax(v_x/alpha, axis=1)
    mu = tf.argmax(pi_log, axis=1)

    # tf.random.multinomial( logits, num_samples, seed=None, name=None, output_dtype=None )
    # logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized log-probabilities for all classes.
    # num_samples: 0-D. Number of independent samples to draw for each row slice.
    pi = tf.squeeze(tf.random.multinomial(pi_log, 1), axis=1)

    # logp_pi = tf.reduce_sum(tf.one_hot(mu, depth=act_dim) * pi_log, axis=1)  # use max Q(s,a)
    # logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * pi_log, axis=1)
    logp_pi = tf.reduce_sum(tf.exp(pi_log)*pi_log, axis=1)                     # exact entropy

    return mu, pi, logp_pi



"""
Actor-Critics
"""

def mlp_actor_critic(x, x2,  a, alpha, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=None, policy=softmax_policy, action_space=None):

    if x.shape[1] == 128:                # for Breakout-ram-v4
        x = (x - 128.0) / 128.0          # x: shape(?,128)

    act_dim = action_space.n
    a_one_hot = tf.one_hot(a[...,0], depth=act_dim)      # shape(?,4)
    #vfs
    vf_mlp = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, None)     # return: shape(?,4)


    ################# Q1

    q1_tp = tf.make_template('q1', vf_mlp, create_scope_now_=True)

    v1_x = q1_tp(x)

    # policy
    mu, pi, logp_pi = policy(alpha, v1_x, act_dim)

    mu_one_hot = tf.one_hot(mu, depth=act_dim)
    pi_one_hot = tf.one_hot(pi, depth=act_dim)

    q1 = tf.reduce_sum(v1_x * a_one_hot, axis=1)

    q1_mu = tf.reduce_sum(v1_x * mu_one_hot, axis=1)  # use max Q(s,a)
    q1_pi = tf.reduce_sum(v1_x * pi_one_hot, axis=1)

    v1_x2 = q1_tp(x2)

    # policy
    mu_x2, pi_x2, logp_pi_x2 = policy(alpha, v1_x2, act_dim)


    ################# Q2

    q2_tp = tf.make_template('q2', vf_mlp, create_scope_now_=True)

    v2_x = q2_tp(x)

    mu2, pi2, logp_pi2 = policy(alpha, v2_x, act_dim)

    mu2_one_hot = tf.one_hot(mu2, depth=act_dim)

    q2 = tf.reduce_sum(v2_x * a_one_hot, axis=1)

    q2_mu = tf.reduce_sum(v2_x * mu2_one_hot, axis=1)  # use max Q(s,a)
    q2_pi = tf.reduce_sum(v2_x * pi_one_hot, axis=1)

    # shape(?,)
    return mu, pi, logp_pi, logp_pi_x2, q1, q2, q1_pi, q2_pi, q1_mu, q2_mu


