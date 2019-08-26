import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import time
from spinup.algos.sac1_rnn import core
from spinup.algos.sac1_rnn.core import get_vars
from spinup.utils.logx import EpochLogger
from gym.spaces import Box, Discrete
from spinup.utils.frame_stack import FrameStack
from collections import deque

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import os

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class ReplayBuffer_RNN:
    """
    A simple FIFO experience replay buffer for SAC_RNN agents.
    """

    def __init__(self, Lb, Lt, hc_dim, obs_dim, act_dim, size):
        self.buffer_obs = np.zeros([size, Lb + Lt + 1, obs_dim], dtype=np.float32)
        self.buffer_hc = np.zeros([size, hc_dim], dtype=np.float32)
        self.buffer_a = np.zeros([size, Lb + Lt, act_dim], dtype=np.float32)
        self.buffer_r = np.zeros([size, Lb + Lt], dtype=np.float32)
        self.buffer_d = np.zeros([size, Lb + Lt], dtype=np.float32)
        self.buffer_data01 = np.zeros([size, Lb + Lt], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs_hc_queue, a_r_d_data01_queue):
        obs, hc = np.stack(obs_hc_queue, axis=1)
        self.buffer_obs[self.ptr] = np.array(list(obs), dtype=np.float32)
        self.buffer_hc[self.ptr] = np.array(list(hc), dtype=np.float32)[0]
        a, r, d, data01 = np.stack(a_r_d_data01_queue, axis=1)
        self.buffer_a[self.ptr] = np.array(list(a), dtype=np.float32)
        self.buffer_r[self.ptr] = np.array(list(r), dtype=np.float32)
        self.buffer_d[self.ptr] = np.array(list(d), dtype=np.float32)
        self.buffer_data01[self.ptr] = np.array(list(data01), dtype=np.float32)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.buffer_obs[idxs],
                    hc=self.buffer_hc[idxs],
                    acts=self.buffer_a[idxs],
                    rews=self.buffer_r[idxs],
                    done=self.buffer_d[idxs],
                    data01=self.buffer_data01[idxs], )


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)


 Lt >= Lb > 0 !!!

"""


def sac1_rnn(args, env_fn, actor_critic=core.mlp_actor_critic, sac1_dynamic_rnn=core.sac1_dynamic_rnn,
             ac_kwargs=dict(), seed=0, Lb=10, Lt=10, hc_dim=128, steps_per_epoch=3000, epochs=100,
             replay_size=int(1e5), gamma=0.99, reward_scale=1.0, polyak=0.995, lr=5e-4, alpha=0.2,
             h0=1.0, batch_size=150, start_steps=10000, max_ep_len_train=1000, max_ep_len_test=1000,
             logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for policy/value/alpha learning).

        alpha (float/'auto'): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.) / 'auto': alpha is automated.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn('train'), env_fn('test')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    ######################################
    # Inputs to computation graph
    # x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    #
    # # Main outputs from computation graph
    # with tf.variable_scope('main'):
    #     mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    #
    # # Target value network
    # with tf.variable_scope('target'):
    #     _, _, logp_pi_, _, _, q1_pi_, q2_pi_ = actor_critic(x2_ph, a_ph, **ac_kwargs)
    #
    ######################################

    obs_ph, hc_ph = core.placeholders((Lb + Lt + 1, obs_dim), (hc_dim,))
    a_ph_all, r_ph_all, d_ph_all, data01_ph = core.placeholders((Lb + Lt, act_dim), (Lb + Lt,), (Lb + Lt,), (Lb + Lt,))

    obs_burn = obs_ph[:, :Lb]
    obs_train = obs_ph[:, Lb:]

    obs12_train = data01_ph[:, Lb:]
    # obs12_train = tf.transpose(obs12_train, perm=[1, 0])

    a_ph = a_ph_all[:, Lb:]
    r_ph = r_ph_all[:, Lb:]
    d_ph = d_ph_all[:, Lb:]

    _, state_burn_in = sac1_dynamic_rnn(obs_burn, hc_ph)
    state_burn_in = tf.stop_gradient(state_burn_in) * data01_ph[:, 0][..., tf.newaxis]
    s_outputs, _ = sac1_dynamic_rnn(obs_train, state_burn_in)
    s_ph = s_outputs[:, :-1]
    s2_ph = s_outputs[:, 1:]

    logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi = [None, ] * Lt, [None, ] * Lt, [None, ] * Lt, \
                                              [None, ] * Lt, [None, ] * Lt, [None, ] * Lt
    logp_pi_, q1_pi_, q2_pi_ = [None, ] * Lt, [None, ] * Lt, [None, ] * Lt

    for i in range(Lt):
        # Main outputs from computation graph
        with tf.variable_scope('main', reuse=tf.AUTO_REUSE):

    ######################################
            _, _, logp_pi[i], logp_pi2[i], q1[i], q2[i], q1_pi[i], q2_pi[i] = actor_critic(s_ph[:, i],
                                                                                           s2_ph[:, i],
                                                                                           a_ph[:, i],
                                                                                           **ac_kwargs)
        # Target value network
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            _, _, logp_pi_[i], _, _, _, q1_pi_[i], q2_pi_[i] = actor_critic(s2_ph[:, i], s2_ph[:, i], a_ph[:, i],
                                                                            **ac_kwargs)

    logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi = tf.stack(logp_pi, axis=1), tf.stack(logp_pi2, axis=1), \
                            tf.stack(q1, axis=1), tf.stack(q2, axis=1), tf.stack(q1_pi, axis=1), tf.stack(q2_pi, axis=1)
    logp_pi_, q1_pi_, q2_pi_ = tf.stack(logp_pi_, axis=1), tf.stack(q1_pi_, axis=1), tf.stack(q2_pi_, axis=1)

    ######################################

    # Experience buffer
    # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer_rnn = ReplayBuffer_RNN(Lb=Lb, Lt=Lt, hc_dim=hc_dim, obs_dim=obs_dim, act_dim=act_dim,
                                         size=replay_size)
    # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in
    #                    ['main/pi', 'main/q1', 'main/q2', 'rnn'])
    # print(('\nNumber of parameters: \t pi: %d, \t' + 'q1: %d, \t q2: %d, \t rnn: %d\n') % var_counts)
    # print('Number of parameters: \t Total: %d\n' % sum(var_counts))

    ######
    if alpha == 'auto':
        target_entropy = (-np.prod(env.action_space.shape))

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)

        alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))

        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr * h0, name='alpha_optimizer')
        train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
    ######

    # Min Double-Q:
    min_q_pi_ = tf.minimum(q1_pi_, q2_pi_)

    # Targets for Q and V regression
    v_backup = tf.stop_gradient(min_q_pi_ - alpha * logp_pi2)
    q_backup = r_ph + gamma * (1 - d_ph) * v_backup

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(obs12_train * (alpha * logp_pi - q1_pi))
    q1_loss = 0.5 * tf.reduce_mean(obs12_train * (q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean(obs12_train * (q_backup - q2) ** 2)
    value_loss = q1_loss + q2_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    pi_params = get_vars('main/pi')
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=pi_params)

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('rnn')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    if isinstance(alpha, Number):
        step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(alpha),
                    train_pi_op, train_value_op, target_update]
    else:
        step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha,
                    train_pi_op, train_value_op, target_update, train_alpha_op]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Inputs to computation graph
    x_ph_geta, hc_ph_geta, a_ph_geta = core.placeholders((1, obs_dim), hc_dim, act_dim)

    s_geta, hc_geta = sac1_dynamic_rnn(x_ph_geta, hc_ph_geta)
    # Main outputs from computation graph
    with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
        mu, pi, _, _, _, _, _, _ = actor_critic(s_geta[:, 0], s_geta[:, 0], a_ph_geta, **ac_kwargs)

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
    #                       outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})
    saver = tf.train.Saver()

    checkpoint_path = logger_kwargs['output_dir'] + '/checkpoints'
    if not os.path.exists(checkpoint_path):
         os.makedirs(checkpoint_path)

    if args.is_test or args.is_restore_train:
         ckpt = tf.train.get_checkpoint_state(checkpoint_path)
         if ckpt and ckpt.model_checkpoint_path:
             saver.restore(sess, ckpt.model_checkpoint_path)
             print("Model restored.")
    # def get_action(o, deterministic=False):
    #     act_op = mu if deterministic else pi
    #     return sess.run(act_op, feed_dict={x_ph_geta: o.reshape(1, -1)})[0]#[0]

    def get_action(o, hc_0, deterministic=False):
        """s_t_0_  starting step for testing 1 H"""

        act_op = mu if deterministic else pi
        action, hc_1 = sess.run([act_op, hc_geta], feed_dict={x_ph_geta: o.reshape(1, 1, obs_dim),
                                                              hc_ph_geta: hc_0})
        return action[0], hc_1
        ##############################  test  ############################

    if args.is_test:
        test_env = gym.make(args.env)
        ave_ep_ret = 0
        for j in range(10000):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not d:  # (d or (ep_len == 2000)):
                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
                if args.test_render:
                    test_env.render()
            ave_ep_ret = (j * ave_ep_ret + ep_ret) / (j + 1)
            print('ep_len', ep_len, 'ep_ret:', ep_ret, 'ave_ep_ret:', ave_ep_ret, '({}/10000)'.format(j + 1))
        return

        ##############################  train  ############################
    def test_agent(n=5):
        # print('test')
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            hc_run_test = np.zeros((1, hc_dim,), dtype=np.float32)
            while not (d or (ep_len == max_ep_len_test)):
                # Take deterministic actions at test time
                a_test, hc_run_test = get_action(o, hc_run_test, True)
                o, r, d, _ = test_env.step(a_test)
                ep_ret += r
                ep_len += 1
                # test_env.render()
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    ################################## deques

    obs_hc_queue = deque([], maxlen=Lb + Lt + 1)
    a_r_d_data01_queue = deque([], maxlen=Lb + Lt)

    ################################## deques

    start_time = time.time()

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    ################################## deques reset
    t_queue = 1
    hc_run = np.zeros((1, hc_dim,), dtype=np.float32)
    for _i in range(Lb):
        obs_hc_queue.append((np.zeros((obs_dim,), dtype=np.float32), np.zeros((hc_dim,), dtype=np.float32)))
        a_r_d_data01_queue.append((np.zeros((act_dim,), dtype=np.float32), 0.0, False, False))
    obs_hc_queue.append((o, hc_run[0]))

    ################################## deques reset

    total_steps = steps_per_epoch * epochs

#    test_ep_ret = test_ep_ret_1 = -10000.0
    test_ep_ret_best = test_ep_ret = -10000.0
    # Main loop: collect experience in env and update/log each epoch
    start = time.time()
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a, hc_run = get_action(o, hc_run)
        else:
            _, hc_run = get_action(o, hc_run)
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len==max_ep_len_train else d

        # Store experience to replay buffer
        # replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        #################################### deques store

        a_r_d_data01_queue.append((a, r, d, True))
        obs_hc_queue.append((o2, hc_run[0]))

        if t_queue % Lt == 0:
            replay_buffer_rnn.store(obs_hc_queue, a_r_d_data01_queue)

        if (d or (ep_len == max_ep_len_train)) and t_queue % Lt != 0:
            for _0 in range(Lt - t_queue % Lt):
                a_r_d_data01_queue.append((np.zeros((act_dim,), dtype=np.float32), 0.0, False, False))
                obs_hc_queue.append((np.zeros((obs_dim,), dtype=np.float32), np.zeros((hc_dim,), dtype=np.float32)))
            replay_buffer_rnn.store(obs_hc_queue, a_r_d_data01_queue)

        t_queue += 1

        #################################### deques store

        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len_train):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer_rnn.sample_batch(batch_size)
                feed_dict = {obs_ph: batch['obs'],
                             hc_ph: batch['hc'],
                             a_ph_all: batch['acts'],
                             r_ph_all: batch['rews'],
                             d_ph_all: batch['done'],
                             data01_ph: batch['data01']
                             }
                # step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha, train_pi_op, train_value_op, target_update]
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0],
                             LossQ1=outs[1],
                             LossQ2=outs[2],
                             Q1Vals=outs[3][:, 0],
                             Q2Vals=outs[4][:, 0],
                             LogPi=outs[5][:, 0],
                             Alpha=outs[6])

            logger.store(EpRet=ep_ret / reward_scale, EpLen=ep_len)
            print("ep_len", ep_len, "time", time.time() - start)
            start = time.time()
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            ################################## deques reset
            t_queue = 1
            hc_run = np.zeros((1, hc_dim,), dtype=np.float32)
            for _i in range(Lb):
                obs_hc_queue.append((np.zeros((obs_dim,), dtype=np.float32), np.zeros((hc_dim,), dtype=np.float32)))
                a_r_d_data01_queue.append((np.zeros((act_dim,), dtype=np.float32), 0.0, False, False))
            obs_hc_queue.append((o, hc_run[0]))

            ################################## deques reset

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            if epoch < 2000:
                test_agent(25)
                # test_ep_ret = logger.get_stats('TestEpRet')[0]
                # print('TestEpRet', test_ep_ret, 'Best:', test_ep_ret_best)
            else:
                test_agent(25)
                test_ep_ret = logger.get_stats('TestEpRet')[0]
                # logger.epoch_dict['TestEpRet'] = []
                print('TestEpRet', test_ep_ret, 'Best:', test_ep_ret_best)
            # test_agent(25)

            # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Name', name)
            logger.log_tabular('EpRet', with_min_and_max=True)

            logger.log_tabular('TestEpRet', with_min_and_max=True)
            # test_ep_ret_1 = logger.get_stats('TestEpRet')[0]

            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=False)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            # Save model
            if ((epoch % save_freq == 0) or (epoch == epochs - 1)) and test_ep_ret > test_ep_ret_best:
                save_path = saver.save(sess, checkpoint_path + '/model.ckpt', t)
                print("Model saved in path: %s" % save_path)
                test_ep_ret_best = test_ep_ret


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='BipedalWalkerHardcore-v2')  # 'Pendulum-v0'

    # parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--env', type=str, default='Pendulum-v0')  # 'Pendulum-v0'
    parser.add_argument('--is_restore_train', type=bool, default=False)
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--test_render', type=bool, default=False)
    parser.add_argument('--max_ep_len_test', type=int, default=2000)  # 'BipedalWalkerHardcore-v2' max_ep_len is 2000
    parser.add_argument('--max_ep_len_train', type=int,
                        default=400)  # max_ep_len_train < 2000//3 # 'BipedalWalkerHardcore-v2' max_ep_len is 2000
    parser.add_argument('--h1', type=int, default=256)
    parser.add_argument('--h2', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--alpha', default="auto", help="alpha can be either 'auto' or float(e.g:0.2).")

    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--act_noise', type=float, default=0.0)
    parser.add_argument('--obs_noise', type=float, default=0.0)
    # parser.add_argument('--exp_name', type=str, default='sac1_rnn_BipedalWalkerHardcore-v2_debug')
    parser.add_argument('--act_repeate', type=int, default=1)
    parser.add_argument('--Lt', type=int, default=15)  # 'train'
    parser.add_argument('--Lb', type=int, default=10)  # 'burn-in'
    parser.add_argument('--hc_dim', type=int, default=64)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--h0', type=float, default=0.1)  # for alpha learning rate decay
    # parser.add_argument('--epochs', type=int, default=1000)
    # parser.add_argument('--alpha', default="auto", help="alpha can be either 'auto' or float(e.g:0.2).")
    name = 'debug_sac1_rnn_{}_Lt_{}_h0_{}_alpha_{}_seed_{}'.format(
        parser.parse_args().env,
        parser.parse_args().Lt,
        # parser.parse_args().hid1,
        # parser.parse_args().hid2,
        # parser.parse_args().state,
        # parser.parse_args().flag,
        parser.parse_args().h0,
        parser.parse_args().alpha,
        parser.parse_args().seed)
    # parser.parse_args().beta,
    # parser.parse_args().tm,
    # parser.parse_args().repeat,
    # parser.parse_args().ps)
    parser.add_argument('--exp_name', type=str, default=name)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    class Wrapper_train(object):

        def __init__(self, env, action_repeat=3):
            self._env = env
            self.action_repeat = action_repeat
            self.obs_dim = env.observation_space.shape[0]
            self.act_dim = env.action_space.shape[0]

        def __getattr__(self, name):
            return getattr(self._env, name)

        def reset(self):
            obs = self._env.reset() + args.obs_noise * (-2 * np.random.random(self.obs_dim) + 1)
            return obs

        def step(self, action):
            action += args.act_noise * (-2 * np.random.random(self.act_dim) + 1)
            r = 0.0
            for _ in range(self.action_repeat):
                obs_, reward_, done_, info_ = self._env.step(action)
                r = r + reward_
                # r -= 0.001
                if done_:
                    return obs_ + args.obs_noise * (-2 * np.random.random(self.obs_dim) + 1), 0.0, done_, info_

            return obs_ + args.obs_noise * (-2 * np.random.random(self.obs_dim) + 1), args.reward_scale * r, done_, info_


    class Wrapper_test(object):

        def __init__(self, env, action_repeat=3):
            self._env = env
            self.action_repeat = action_repeat

        def __getattr__(self, name):
            return getattr(self._env, name)

        def step(self, action):
            r = 0.0
            for _ in range(self.action_repeat):
                obs_, reward_, done_, info_ = self._env.step(action)
                r = r + reward_
                # r -= 0.001
                if done_:
                    return obs_, r, done_, info_

            return obs_, r, done_, info_


    # env = FrameStack(env, args.stack_frames)

    env_train = Wrapper_train(gym.make(args.env), args.act_repeate)
    env_test = Wrapper_test(gym.make(args.env), args.act_repeate)

    sac1_rnn(args,
             lambda x: env_train if x == 'train' else env_test,
             actor_critic=core.mlp_actor_critic,
             sac1_dynamic_rnn=core.sac1_dynamic_rnn1,
             ac_kwargs=dict(hidden_sizes=[args.h1, args.h2]),
             Lb=args.Lb,
             Lt=args.Lt,
             hc_dim=args.hc_dim,
             gamma=args.gamma,
             seed=args.seed,
             epochs=args.epochs,
             alpha=args.alpha,
             h0=args.h0,
             logger_kwargs=logger_kwargs,
             lr=args.lr,
             reward_scale=args.reward_scale,
             max_ep_len_train=args.max_ep_len_train,
             max_ep_len_test=args.max_ep_len_test)
