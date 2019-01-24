import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import time
from spinup.algos.sqn_rpf import core
from spinup.algos.sqn_rpf.core import get_vars
from spinup.utils.logx import EpochLogger
from gym.spaces import Box, Discrete



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)




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
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sqn_rpf(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1, ensemble_size=10):
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
    # print(max_ep_len,type(max_ep_len))
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)


    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    obs_space = env.observation_space
    act_dim = env.action_space.n
    act_space = env.action_space


    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders_from_space(obs_space, act_space, obs_space, None, None)
    # x_ph, x2_ph: shape(?,128)
    # a_ph: shape(?,1)
    # r_ph, d_ph: shape(?,)


    ######
    if alpha == 'auto':
        # target_entropy = (-np.prod(env.action_space.n))
        # target_entropy = (np.prod(env.action_space.n))/4/10
        target_entropy = 0.15

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)
    ######


    # Main outputs from computation graph
    with tf.variable_scope('random_head'):
        head_index = tf.get_variable(name='random_int', shape=[], dtype=tf.int32)

    with tf.variable_scope('main'):
        mu, pi, _, q1, _, q2, _ = actor_critic(x_ph, a_ph, alpha, ensemble_size=ensemble_size, **ac_kwargs)
        # _, _, logp_pi, _, _ = actor_critic(x2_ph, a_ph, alpha, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, logp_pi_, _, q1_pi_, _, q2_pi_= actor_critic(x2_ph, a_ph, alpha, ensemble_size=ensemble_size, **ac_kwargs)

    # Experience buffer
    if isinstance(act_space, Box):
        a_dim = act_dim
    elif isinstance(act_space, Discrete):
        a_dim = 1
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=a_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t total: %d\n')%var_counts)


######
    if isinstance(alpha,tf.Tensor):
        alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi_ + target_entropy))

        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='alpha_optimizer')
        train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
######

    # Min Double-Q:
    min_q_pi = [tf.minimum(q1_pi_[i], q2_pi_[i]) for i in range(ensemble_size)]

    # Targets for Q and V regression
    # v_backup = tf.stop_gradient(q1_pi_ - alpha * logp_pi_)  ############################## alpha=0
    v_backup = [tf.stop_gradient(min_q_pi[i] - alpha * logp_pi_[i]) for i in range(ensemble_size)]
    # q_backup = tf.expand_dims(r_ph, axis=-1) + gamma*(1-tf.expand_dims(d_ph, axis=-1))*v_backup
    # q_backup = r_ph + gamma * (1 - d_ph) * v_backup
    q_backup = [r_ph + gamma * (1 - d_ph) * v_backup[i]  for i in range(ensemble_size)]

    # Soft actor-critic losses
    # q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    # q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    # value_loss = q1_loss + q2_loss
    q1_loss = [0.5 * tf.reduce_mean((q_backup[i] - q1[i])**2, axis=0) for i in range(ensemble_size)]
    q2_loss = [0.5 * tf.reduce_mean((q_backup[i] - q2[i])**2, axis=0) for i in range(ensemble_size)]
    value_loss = [q1_loss[i] + q2_loss[i] for i in range(ensemble_size)]


    # # Policy train op
    # # (has to be separate from value train op, because q1_pi appears in pi_loss)
    # pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    #with tf.control_dependencies([train_pi_op]):
    train_value_op = [value_optimizer.minimize(value_loss[i], var_list=value_params) for i in range(ensemble_size)]
    # train_value_op = [value_optimizer.minimize(value_loss)]

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies(train_value_op):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])         # zip([1,2,3,4],['a','b']) = [(1,'a'),(2,'b')]

    target_update_1 = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])         # zip([1,2,3,4],['a','b']) = [(1,'a'),(2,'b')]
    step_ops_1 = [q1_loss[0], q1[0], logp_pi_[0], tf.identity(alpha), target_update_1]


    # All ops to call during one training step
    if isinstance(alpha, Number):
        step_ops = [q1_loss[0], q1[0], logp_pi_[0], tf.identity(alpha), train_value_op, target_update]
    else:
        step_ops = [q1_loss, q1, logp_pi_, alpha, train_value_op, target_update, train_alpha_op]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu[0], 'pi': pi[0], 'q1': q1[0]})

    def get_action(o, active_head=0, deterministic=False):
        act_op = mu[active_head] if deterministic else pi[active_head]
        return sess.run(act_op, feed_dict={x_ph: np.expand_dims(o, axis=0)})[0]

    def test_agent(n=3):  # number of tests
        global sess, mu, pi, q1
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):  # max_ep_len
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, deterministic=True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)



    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


    # Select a head to interact with env.
    active_head = np.random.randint(ensemble_size)

    # t0 = time.time()


    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """

        # if t > start_steps and 20*t/total_steps > np.random.random(): # greedy, avoid falling into sub-optimum
        if t > start_steps:
            a = get_action(o, active_head=active_head)
        else:
            a = env.action_space.sample()

        np.random.random()


        # Step the env
        o2, r, d, _ = env.step(a)
        #print(a,o2)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len):

            # t_last = t0
            # t0 = time.time()
            # print('episode_time:', t0-t_last, 'ep_len:', ep_len)

            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):

                if False:
                    ########## scheme 0 ############
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done'],
                                }
                    # step_ops = [q1_loss, q1, logp_pi_, alpha, train_value_op, target_update, train_alpha_op]
                    outs = sess.run(step_ops, feed_dict)
                    ###############################

                else:
                    ########## scheme 1 ############
                    for i in range(ensemble_size):
                        batch = replay_buffer.sample_batch(batch_size)
                        feed_dict = {x_ph: batch['obs1'],
                                     x2_ph: batch['obs2'],
                                     a_ph: batch['acts'],
                                     r_ph: batch['rews'],
                                     d_ph: batch['done'],
                                    }
                        sess.run(train_value_op[i], feed_dict)
                    # step_ops_1 = [q1_loss, q1, logp_pi_, alpha, target_update, train_alpha_op]
                    outs = sess.run(step_ops_1, feed_dict)
                    ###############################


                logger.store(LossQ1=outs[0], Q1Vals=outs[1],
                            LogPi=outs[2], Alpha=outs[3])

            logger.store(EpRet=ep_ret, EpLen=ep_len)

            # t_last = t0
            # t0 = time.time()
            # print('training_time:', t0-t_last, 'num_train/ep_len:', ep_len)


            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            # Select a head to interact with env.
            active_head = np.random.randint(ensemble_size)
            # print(active_head)

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:  # and ep_len < steps_per_epoch:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Alpha',average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            # logger.log_tabular('LossQ2', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')  # CartPole-v0 Acrobot-v1 LunarLander-v2 Breakout-ram-v4 MountainCar-v0 Atlantis-ram-v0
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--max_ep_len', type=int, default=1000)    # make sure: max_ep_len < steps_per_epoch
    parser.add_argument('--alpha', type=float, default=2.0, help="alpha can be either 'auto' or float(e.g:0.2).")
    parser.add_argument('--ensemble_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--exp_name', type=str, default='sqn_rpf_test2')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sqn_rpf(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[400,300]), ensemble_size=args.ensemble_size,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, alpha=args.alpha, lr=args.lr, max_ep_len = args.max_ep_len,
        logger_kwargs=logger_kwargs)