import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import time
from spinup.algos.sac1_rnn import core
from spinup.algos.sac1_rnn.core import get_vars, rnn_cell, rnn_gaussian_policy, apply_squashing_func
from spinup.utils.logx import EpochLogger
import gym
from gym.spaces import Box, Discrete, Tuple


# seq_length = 17
# seq_length = 20  # sequence length (timestep) T
# h_size = 256  # hidden state size H
#  = 128  # batch size

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, h_size, seq_length, flag="single"):
        self.flag = flag
        self.sequence_length = seq_length
        self.ptr, self.size, self.max_size = 0, 0, size
        self.obs_dim = obs_dim
        size += seq_length  # in case index is out of range
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.hidden_buf = np.zeros([size, h_size], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.target_done_ratio = 0

    def store(self, obs, s_t_0, act, rew, done):
        self.obs1_buf[self.ptr] = obs
        self.hidden_buf[self.ptr] = s_t_0
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # 1%20=1 2%20=2 21%20=1
        self.size = min(self.size + 1, self.max_size)  # use self.size to control sample range
        self.target_done_ratio = np.sum(self.done_buf) / self.size

    def sample_batch(self, batch_size=32):
        """
        :param batch_size:
        :return: s a r s' d
        """

        idxs_c = np.empty([batch_size, self.sequence_length])  # N T+1

        for i in range(batch_size):
            end = False
            while not end:
                ind = np.random.randint(0, self.size - 5)  # random sample a starting point in current buffer
                idxs = np.arange(ind, ind + self.sequence_length)  # extend seq from starting point
                is_valid_pos = True if sum(self.done_buf[idxs]) == 0 else (self.sequence_length -
                                                                           np.where(self.done_buf[idxs] == 1)[0][
                                                                               0]) == 2

                end = True if is_valid_pos else False

            idxs_c[i] = idxs

        np.random.shuffle(idxs_c)
        idxs = idxs_c.astype(int)
        # print(self.target_done_ratio, np.sum(self.done_buf[idxs]) / batch_size)
        data = dict(obs1=self.obs1_buf[idxs],
                    s_t_0=self.hidden_buf[idxs][:, 0, :],  # slide N T H to N H
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
        return data


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""


def sac1_rnn(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, lr=6e-4, alpha=0.2, batch_size=128, start_steps=10000,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
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
    # print(batch_size)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    h_size = ac_kwargs["h_size"]
    seq_length = ac_kwargs["seq"]

    # Inputs to computation graph
    seq = None
    x_ph, a_ph, r_ph, d_ph = core.placeholders([seq, obs_dim], [seq, act_dim], [seq, 1], [seq, 1])
    s_t_0 = tf.placeholder(shape=[None, h_size], name="pre_state", dtype="float32")  # zero state
    s_0 = np.zeros([batch_size, h_size])  # zero                       state for training  N H

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, states = actor_critic(x_ph, a_ph, s_t_0, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, _, logp_pi_, _, _, q1_pi_, q2_pi_, _ = actor_critic(x_ph, a_ph, s_t_0, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 h_size=h_size, seq_length=seq_length, flag="seq")

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t total: %d\n') % var_counts)

    if alpha == 'auto':
        # target_entropy = (-np.prod(env.action_space.shape))
        target_entropy = -np.prod(env.action_space.shape)

        # log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        print(ac_kwargs["h0"])
        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=ac_kwargs["h0"])
        alpha = tf.exp(log_alpha)

        alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi[:, :-1, :] + target_entropy))
        # Use smaller learning rate to make alpha decay slower
        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name='alpha_optimizer')
        train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

    # Min Double-Q:
    # min_q_pi = tf.minimum(q1_pi_, q2_pi_)

    # Targets for Q and V regression
    v_backup = tf.stop_gradient(tf.minimum(q1_pi_, q2_pi_) - alpha * logp_pi)
    q_backup = r_ph[:, :-1, :] + gamma * (1 - d_ph[:, :-1, :]) * v_backup[:, 1:, :]

    # Soft actor-critic losses
    # pi_loss = tf.reduce_mean(alpha * logp_pi[:, :-1, :] - q1_pi[:, :-1, :])
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    # in some case, the last timestep Q function is super important so maybe we can use weight sum of loss
    # calculate last timestep separately for convince
    q1_loss = 0.5 * tf.reduce_mean((q1[:, :-1, :] - q_backup) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q2[:, :-1, :] - q_backup) ** 2)
    value_loss = q1_loss + q2_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # TODO: maybe we should add parameters in main/rnn to optimizer ---> training is super slow while we adding it
    value_params = get_vars('main/q') + get_vars('main/rnn')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in non_deterministic order)
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

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                          outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, s_t_0_, mu, pi, states, deterministic=False):
        """s_t_0_  starting step for testing 1 H"""

        act_op = mu if deterministic else pi
        action, s_t_1_ = sess.run([act_op, states], feed_dict={x_ph: o.reshape(1, 1, obs_dim),
                                                               a_ph: np.zeros([1, 1, act_dim]),
                                                               s_t_0: s_t_0_})
        return action.reshape(act_dim), s_t_1_

    def test_agent(mu, pi, states, n=5):
        # global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            s_0 = np.zeros([1, h_size])
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a, s_1 = get_action(o, s_0, mu, pi, states, deterministic=True)
                s_0 = s_1
                o, r, d, _ = test_env.step(a)
                # test_env.render()
                ep_ret += r
                ep_len += 1
                # replay_buffer.store(o.reshape([1, obs_dim]), a.reshape([1, act_dim]), r, d)
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    s_t_0_ = np.zeros([1, h_size])
    episode = 0

    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        # if t == 0:
        #     start = time.time()

        if t > start_steps:
            # TODO: try to store hidden state in buffer
            # s_t_0_store = s_t_0_    # hidden state stored in buffer
            a, s_t_1_ = get_action(o, s_t_0_, mu, pi, states, deterministic=False)
            s_t_0_ = s_t_1_
        else:
            # s_t_0_store = s_t_0_
            _, s_t_1_ = get_action(o, s_t_0_, mu, pi, states, deterministic=False)
            s_t_0_ = s_t_1_
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)  # give back o_t_1 we need store o_t_0 because that is what cause a_t_0
        # print(r)
        # env.render()
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o.reshape([1, obs_dim]), s_t_0_.reshape([1, h_size]), a.reshape([1, act_dim]), r, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            # fps = (time.time() - start)/200
            # print("{} fps".format(200 / (time.time() - start)))
            print(ep_len)
            episode += 1
            for j in range(int(ep_len)):
                batch = replay_buffer.sample_batch(batch_size)
                # maybe we can store starting state
                feed_dict = {x_ph: batch['obs1'],
                             s_t_0: batch['s_t_0'],  # all zero matrix for zero state in training
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             }
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             Q1Vals=outs[3], Q2Vals=outs[4],
                             LogPi=outs[5], Alpha=outs[6])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            s_t_0_ = np.zeros([1, h_size])  # reset s_t_0_ when one episode is finished
            # start = time.time()

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(mu, pi, states)

            # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Episode', episode)
            logger.log_tabular('name', name)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()
            # start = time.time()


if __name__ == '__main__':
    import argparse
    # from OpenGL import GLU
    import argparse
    # import roboschool
    from gym_env import Env_wrapper

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    # parser.add_argument('--env', type=str, default="RoboschoolHalfCheetah-v1")
    parser.add_argument('--env', type=str, default='BipedalWalkerHardcore-v2')
    parser.add_argument('--flag', type=str, default='obs_act')
    parser.add_argument('--hid1', type=int, default=400)
    parser.add_argument('--hid2', type=int, default=300)
    parser.add_argument('--state', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seq', type=int, default=18)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--h0', type=float, default=0.0)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--alpha', default=0.5, help="alpha can be either 'auto' or float(e.g:0.2).")
    name = 'seed_{}_test_{}_seq_{}_mlp_{}_{}_rnn_{}_obs_{}_h0_{}_alpha_{}'.format(parser.parse_args().seed,
                                                                                  parser.parse_args().env,
                                                                                  parser.parse_args().seq,
                                                                                  parser.parse_args().hid1,
                                                                                  parser.parse_args().hid2,
                                                                                  parser.parse_args().state,
                                                                                  parser.parse_args().flag,
                                                                                  parser.parse_args().h0,
                                                                                  parser.parse_args().alpha)
    parser.add_argument('--exp_name', type=str, default=name)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    class Wrapper(object):

        def __init__(self, env):
            self._env = env

        def __getattr__(self, name):
            return getattr(self._env, name)

        def reset(self):
            obs = self._env.reset()
            return obs

        def step(self, action):
            r = 0.0
            for _ in range(3):
                obs_, reward_, done_, info_ = self._env.step(action)
                r = r + reward_
            return obs_, r, done_, info_

    sac1_rnn(lambda: Env_wrapper(args.env, args.flag, 3), # lambda:Wrapper(gym.make(args.env)), #
         actor_critic=core.rnn_actor_critic,
         batch_size=args.batch_size,
         ac_kwargs=dict(hidden_sizes=[128,128,128], h_size=args.state, seq=args.seq, h0=args.h0),
         gamma=args.gamma,
         seed=args.seed,
         epochs=args.epochs,
         alpha=args.alpha,
         logger_kwargs=logger_kwargs)
