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






# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





def mlp_ensemble_with_prior(x, hidden_sizes=(32,), ensemble_size=10, prior_scale=1.0, activation=None, output_activation=None):

    # An ensemble of Prior nets.
    priors = []
    for _ in range(ensemble_size):
        x_proxy = x
        for h in hidden_sizes[:-1]:
            x_proxy = tf.layers.dense(x_proxy, units=h, activation=activation, kernel_initializer=tf.variance_scaling_initializer(2.0))
        priors.append(tf.stop_gradient(tf.layers.dense(x_proxy, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=tf.variance_scaling_initializer(2.0))))      # outputs: 10 x shape(?, 4)

    prior_nets = priors                             # 10 x shape(?, 4)

    # An ensemble of Q nets.
    qs = []
    for _ in range(ensemble_size):
        x_proxy = x
        for h in hidden_sizes[:-1]:
            x_proxy = tf.layers.dense(x_proxy, units=h, activation=activation, kernel_initializer=tf.variance_scaling_initializer(2.0))
        qs.append(tf.layers.dense(x_proxy, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=tf.variance_scaling_initializer(2.0)))

    q_nets = qs

    # An ensemble of Q models.
    q_models = [q_nets[i] + prior_scale * prior_nets[i] for i in range(ensemble_size)]

    return q_models



"""
Policies
"""

def softmax_policy(alpha, v_x, act_dim):

    pi_log = tf.nn.log_softmax(v_x/alpha, axis=1)
    mu = tf.argmax(pi_log, axis=1)

    # tf.random.multinomial( logits, num_samples, seed=None, name=None, output_dtype=None )
    # logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized log-probabilities for all classes.
    # num_samples: 0-D. Number of independent samples to draw for each row slice.
    pi = tf.squeeze(tf.random.multinomial(pi_log, num_samples=1), axis=1)

    # logp_pi = tf.reduce_sum(tf.one_hot(mu, depth=act_dim) * pi_log, axis=1)  # use max Q(s,a)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * pi_log, axis=1)
    # logp_pi = tf.reduce_sum(tf.exp(pi_log)*pi_log, axis=1)                     # exact entropy

    return mu, pi, logp_pi



"""
Actor-Critics
"""

def mlp_actor_critic(x, a, alpha, hidden_sizes=(400,300), ensemble_size=10, activation=tf.nn.relu,
                     output_activation=None, policy=softmax_policy, action_space=None):

    if x.shape[1] == 128:                # for Breakout-ram-v4
        x = (x - 128.0) / 128.0          # x: shape(?,128)

    act_dim = action_space.n
    a_one_hot = tf.squeeze(tf.one_hot(a, depth=act_dim), axis=1)      # shape(?,4)


    #vfs
    # vf_mlp = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, None)     # return: shape(?,4)
    vf_mlp = lambda x: mlp_ensemble_with_prior(x, list(hidden_sizes) + [act_dim], ensemble_size=ensemble_size, activation=activation, output_activation=output_activation)

    with tf.variable_scope('q1'):
        vx1_a = vf_mlp(x)
        q1 = [tf.reduce_sum(vx1_a[i]*a_one_hot, axis=1) for i in range(ensemble_size)]

    with tf.variable_scope('q1', reuse=True):
        vx1_b= vf_mlp(x)
        # policy
        mu, pi, logp_pi = [], [], []
        for i in range(ensemble_size):
            mu_pi_logpi = policy(alpha, vx1_b[i], act_dim)
            mu.append(mu_pi_logpi[0])
            pi.append(mu_pi_logpi[1])
            logp_pi.append(mu_pi_logpi[2])

        # mu_one_hot = tf.one_hot(mu, depth=act_dim)
        pi_one_hot = [tf.one_hot(pi[i], depth=act_dim) for i in range(ensemble_size)]

        # q1_pi = tf.reduce_sum(v_x*mu_one_hot, axis=1)   # use max Q(s,a)
        q1_pi = [tf.reduce_sum(vx1_b[i] * pi_one_hot[i], axis=1) for i in range(ensemble_size)]

    with tf.variable_scope('q2'):
        vx2_a = vf_mlp(x)
        q2 = [tf.reduce_sum(vx2_a[i]*a_one_hot, axis=1) for i in range(ensemble_size)]
    with tf.variable_scope('q2', reuse=True):
        vx2_b = vf_mlp(x)
        # q2_pi = tf.reduce_sum(vf_mlp(x)*mu_one_hot, axis=1)   # use max Q(s,a)
        q2_pi = [tf.reduce_sum(vx2_b[i] * pi_one_hot[i], axis=1) for i in range(ensemble_size)]

    # 10 x shape(?,)
    return mu, pi, logp_pi, q1, q1_pi, q2, q2_pi



