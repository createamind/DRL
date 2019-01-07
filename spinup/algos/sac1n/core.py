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




"""
Policies
"""

def softmax_policy(alpha, v_x, act_dim):

    mu = tf.argmax(v_x, 1)
    v_softmax = tf.nn.softmax(alpha*v_x)
    pi = tf.squeeze(tf.random.multinomial(v_softmax, 1), axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * tf.log(v_softmax), axis=1)

    return mu, pi, logp_pi

"""
Actor-Critics
"""

def mlp_actor_critic(x, a, alpha, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=None, policy=softmax_policy, action_space=None):

    act_dim = action_space.n
    a_one_hot = tf.one_hot(a[...,0], depth=act_dim)

    #vfs
    vf_mlp = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, None)

    with tf.variable_scope('q1'):
        q1 = tf.reduce_sum(vf_mlp(x)*a_one_hot, axis=1)

    with tf.variable_scope('q1', reuse=True):
        v_x= vf_mlp(x)
        # policy
        mu, pi, logp_pi = policy(alpha, v_x, act_dim)
        pi_one_hot = tf.one_hot(pi, depth=act_dim)

        q1_pi = tf.reduce_sum(v_x*pi_one_hot, axis=1)

    with tf.variable_scope('q2'):
        q2 = tf.reduce_sum(vf_mlp(x)*a_one_hot, axis=1)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = tf.reduce_sum(vf_mlp(x)*pi_one_hot, axis=1)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi



