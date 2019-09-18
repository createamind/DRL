import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import time, os
from spinup.algos.maxsqn import core
from spinup.algos.maxsqn.core import get_vars
from spinup.utils.logx import EpochLogger
from gym.spaces import Box, Discrete
from collections import deque

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



class ReplayBuffer_N:
    """
    A simple FIFO experience replay buffer for SAC_N_STEP agents.
    """

    def __init__(self, Ln, obs_shape, act_shape, size):
        self.buffer_o = np.zeros((size, Ln + 1)+obs_shape, dtype=np.float32)
        self.buffer_a = np.zeros((size, Ln)+act_shape, dtype=np.float32)
        self.buffer_r = np.zeros((size, Ln), dtype=np.float32)
        self.buffer_d = np.zeros((size, Ln), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, o_queue, a_r_d_queue):
        obs, = np.stack(o_queue, axis=1)
        self.buffer_o[self.ptr] = np.array(list(obs), dtype=np.float32)

        a, r, d, = np.stack(a_r_d_queue, axis=1)
        self.buffer_a[self.ptr] = np.array(list(a), dtype=np.float32)
        self.buffer_r[self.ptr] = np.array(list(r), dtype=np.float32)
        self.buffer_d[self.ptr] = np.array(list(d), dtype=np.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.buffer_o[idxs],
                    acts=self.buffer_a[idxs],
                    rews=self.buffer_r[idxs],
                    done=self.buffer_d[idxs],)



"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""

""" make sure: max_ep_len < steps_per_epoch """

def maxsqn(args, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(5e6), gamma=0.99, Ln=3,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=20000,
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
    if not args.is_test:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)


    env, test_env = env_fn(3), env_fn(1)

    # # gym env
    # obs_dim = env.observation_space.shape[0]
    # obs_space = env.observation_space
    # google football
    scenario_obsdim = {'academy_empty_goal':32, 'academy_empty_goal_random':32, 'academy_3_vs_1_with_keeper':44, 'academy_3_vs_1_with_keeper_random':44, 'academy_single_goal_versus_lazy':108}
    scenario_obsdim['academy_single_goal_versus_lazy'] = 108
    scenario_obsdim['academy_single_goal_versus_lazy_random'] = 108
    scenario_obsdim['11_vs_11_stochastic']= 108
    scenario_obsdim['11_vs_11_stochastic_random'] = 108
    obs_dim = scenario_obsdim[args.env]
    obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)


    o_shape = obs_space.shape


    act_dim = env.action_space.n
    act_space = env.action_space   # Discrete(21) for gfootball
    a_shape = act_space.shape      # ()


    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # # a_dim
    # if isinstance(act_space, Box):
    #     a_dim = act_dim
    # elif isinstance(act_space, Discrete):
    #     a_dim = 1


    # Inputs to computation graph
    # x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders_from_space(obs_space, act_space, obs_space, None, None)
    # global x2_ph
    x_ph, a_ph, x2_ph,  = core.placeholders(o_shape, a_shape, o_shape)
    r_ph, d_ph, logp_pi_ph = core.placeholders((Ln,), (Ln,), (Ln,))

    ######
    if alpha == 'auto':
        # target_entropy = (-np.prod(env.action_space.n))
        # target_entropy = (np.prod(env.action_space.n))/4/10
        target_entropy = args.target_entropy

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)
    ######


    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi, q1_mu, q2_mu = actor_critic(x_ph, x2_ph, a_ph, alpha, **ac_kwargs)
    # Target value network
    with tf.variable_scope('target'):
        _, _, logp_pi_, _,  _, _,q1_pi_, q2_pi_,q1_mu_, q2_mu_= actor_critic(x2_ph, x2_ph, a_ph, alpha, **ac_kwargs)


    # Experience buffer
    replay_buffer_nstep= ReplayBuffer_N(Ln=Ln, obs_shape=o_shape, act_shape=a_shape, size=replay_size)

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
    if args.use_max:
        min_q_pi = tf.minimum(q1_mu_, q2_mu_)
    else:
        min_q_pi = tf.minimum(q1_pi_, q2_pi_)        # x2

    min_q_pi = tf.clip_by_value(min_q_pi, -300.0, 900.0)


    # Targets for Q and V regression
    # v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi2)  ############################## alpha=0
    # q_backup = r_ph + gamma*(1-d_ph)*v_backup

    #### n-step backup
    q_backup = tf.stop_gradient(min_q_pi)
    for step_i in reversed(range(Ln)):
        q_backup = r_ph[:,step_i] + gamma*(1-d_ph[:,step_i])*(-alpha * logp_pi_ph[:,step_i]   + q_backup)
    ####




    # Soft actor-critic losses
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    value_loss = q1_loss + q2_loss

    # # Policy train op
    # # (has to be separate from value train op, because q1_pi appears in pi_loss)
    # pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    #with tf.control_dependencies([train_pi_op]):
    train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    if isinstance(alpha, Number):
        step_ops = [q1_loss, q2_loss, q1, q2, logp_pi_, tf.identity(alpha),
                train_value_op, target_update]
        step_ops_notraining = [q1_loss, q2_loss, q1, q2, logp_pi_, tf.identity(alpha)]
    else:
        step_ops = [q1_loss, q2_loss, q1, q2, logp_pi_, alpha,
                train_value_op, target_update, train_alpha_op]
        step_ops_notraining = [q1_loss, q2_loss, q1, q2, logp_pi_, alpha]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)



    ##############################  save and restore  ############################

    saver = tf.train.Saver()

    checkpoint_path = logger_kwargs['output_dir'] + '/checkpoints'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if args.is_test or args.is_restore_train:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored.")



    def get_logp_pi(x):
        logp_pi_s = []
        for Ln_i in range(Ln):
            logp_pi_s.append( sess.run(logp_pi2, feed_dict={x2_ph: x[:,Ln_i+1]}) )
        batch_logp_pi = np.stack(logp_pi_s, axis=1)    # or np.swapaxes(np.array(entropy), 0, 1)
        return batch_logp_pi


    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: np.expand_dims(o, axis=0)})[0]



    ##############################  test  ############################

    if args.is_test:
        test_env = football_env.create_environment(env_name=args.env, representation='simple115', render=True)
        # test_env = gym.make(args.env)
        ave_ep_ret = 0
        for j in range(10000):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == 500)): # (d or (ep_len == 2000)):
                o, r, d, _ = test_env.step(get_action(o, True))
                # print(r, d)
                time.sleep(0.05)
                ep_ret += r
                ep_len += 1
                if args.test_render:
                    test_env.render()
            ave_ep_ret = (j*ave_ep_ret + ep_ret)/(j+1)
            print('ep_len', ep_len, 'ep_ret:', ep_ret, 'ave_ep_ret:',ave_ep_ret,'({}/10000)'.format(j+1) )
        return


    ##############################  train  ############################

    def test_agent(n=20):  # n: number of tests
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):  # max_ep_len
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, args.test_determin))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)



    ################################## deques

    o_queue = deque([], maxlen=Ln + 1)
    a_r_d_queue = deque([], maxlen=Ln)

    ################################## deques


    start_time = time.time()


    # o = env.reset()                                                     #####################
    # o, r, d, ep_ret, ep_len = env.step(1)[0], 0, False, 0, 0            #####################
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


    ################################## deques reset
    t_queue = 1
    o_queue.append((o,))

    ################################## deques reset

    total_steps = steps_per_epoch * epochs

    ep_index = 0
    test_ep_ret_best = test_ep_ret = -10000.0
    is_wrap = False
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        # if t > start_steps and 100*t/total_steps > np.random.random(): # greedy, avoid falling into sub-optimum
        if t > start_steps or args.is_restore_train:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # if np.random.random() > t/(3e6):
        #     a = env.action_space.sample()
        # else:
        #     a = get_action(o)



        # Step the env
        o2, r, d, _ = env.step(a)
        #print(a,o2)
        # o2, r, _, d = env.step(a)                     #####################
        # d = d['ale.lives'] < 5                        #####################

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # d_store = True if r == 1.0 else False

        # Store experience to replay buffer
        # replay_buffer.store(o, a, r, o2, d)
        # print(a,r,d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2


        #################################### deques store

        a_r_d_queue.append( (a, r, d,) )
        o_queue.append((o2,))

        if t_queue % Ln == 0:
            replay_buffer_nstep.store(o_queue, a_r_d_queue)

        if d and t_queue % Ln != 0:
            for _0 in range(Ln - t_queue % Ln):
                a_r_d_queue.append((np.zeros(a_shape, dtype=np.float32), 0.0, True,))
                o_queue.append((np.zeros((obs_dim,), dtype=np.float32), ))
            replay_buffer_nstep.store(o_queue, a_r_d_queue)

        t_queue += 1

        #################################### deques store




        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len):   # make sure: max_ep_len < steps_per_epoch
            ep_index += 1
            print('episode: {}, ep_len: {}, reward: {}'.format(ep_index, ep_len, ep_ret))
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer_nstep.sample_batch(batch_size)
                batch_logp_pi = get_logp_pi(batch['obs'])
                feed_dict = {x_ph: batch['obs'][:,0],
                             x2_ph: batch['obs'][:,-1],
                             a_ph: batch['acts'][:,0],
                             logp_pi_ph: batch_logp_pi,
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                # step_ops = [q1_loss, q2_loss, q1, q2, logp_pi, alpha, train_pi_op, train_value_op, target_update]
                if  t > start_steps:
                    outs = sess.run(step_ops, feed_dict)
                else:
                    outs = sess.run(step_ops_notraining, feed_dict)

                logger.store(LossQ1=outs[0], LossQ2=outs[1],
                            Q1Vals=outs[2], Q2Vals=outs[3],
                            LogPi=outs[4], Alpha=outs[5])
            logger.store(EpRet=ep_ret, EpLen=ep_len)




            # End of epoch wrap-up
            if is_wrap:
                epoch = t // steps_per_epoch

                # if epoch < 30:
                #     test_agent(1)
                #     # test_ep_ret = logger.get_stats('TestEpRet')[0]
                #     # print('TestEpRet', test_ep_ret, 'Best:', test_ep_ret_best)
                # else:
                #     test_agent(1)
                #     test_ep_ret = logger.get_stats('TestEpRet')[0]
                #     # if test_ep_ret > test_ep_ret_best:
                #     #     test_agent(30)
                #     #     test_ep_ret = logger.get_stats('TestEpRet')[0]
                #     print('TestEpRet', test_ep_ret, 'Best:', test_ep_ret_best)

                # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                # logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                # logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Alpha',average_only=True)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                # logger.log_tabular('VVals', with_min_and_max=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                # logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ1', average_only=True)
                logger.log_tabular('LossQ2', average_only=True)
                # logger.log_tabular('LossV', average_only=True)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()


                # Save model
                if ((epoch % args.save_freq == 0) or (epoch == epochs - 1)): # or test_ep_ret > test_ep_ret_best:
                    save_path = saver.save(sess, checkpoint_path+'/model.ckpt', t)
                    print("Model saved in path: %s" % save_path)
                    test_ep_ret_best = test_ep_ret

                is_wrap = False


            # o = env.reset()                                              #####################
            # o, r, d, ep_ret, ep_len = env.step(1)[0], 0, False, 0, 0     #####################
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            ################################## deques reset
            t_queue = 1
            o_queue.append((o,))

            ################################## deques reset



        if t > 0 and t % steps_per_epoch == 0:
            is_wrap = True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #  {'academy_empty_goal':32, 'academy_3_vs_1_with_keeper':44, 'academy_single_goal_versus_lazy':108}
    parser.add_argument('--env', type=str, default='11_vs_11_stochastic_random') #'academy_3_vs_1_with_keeper_random')#
    parser.add_argument('--epochs', type=int, default=200000)
    parser.add_argument('--steps_per_epoch', type=int, default=int(5e3))
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--is_restore_train', type=bool, default=False)

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--test_determin', type=bool, default=True)
    parser.add_argument('--test_render', type=bool, default=False)

    # replay_size, steps_per_epoch, batch_size, start_steps, save_freq

    parser.add_argument('--replay_size', type=int, default=int(3e6))
    parser.add_argument('--Ln', type=int, default=2)
    parser.add_argument('--net', type=list, default=[400,600,400,200])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_steps', type=int, default=int(3e4))

    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--seed', '-s', type=int, default=0)  # maxsqn_football100_a 790, maxsqn_football100_b 110

    parser.add_argument('--max_ep_len', type=int, default=400)    # make sure: max_ep_len < steps_per_epoch
    parser.add_argument('--alpha', default='auto', help="alpha can be either 'auto' or float(e.g:0.2).")
    parser.add_argument('--target_entropy', type=float, default=0.4)
    parser.add_argument('--use_max', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--exp_name', type=str, default='lazy_random_incentive_debug') #'lazy_random_incentive')# ')#'1_{}_seed{}-0-half-random_repeat2'.format(parser.parse_args().env,parser.parse_args().seed))
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    import gfootball.env as football_env


    # reward wrapper
    class FootballWrapper(object):

        def __init__(self, env):
            self._env = env
            self.dis_to_goal = 0.0

        def __getattr__(self, name):
            return getattr(self._env, name)

        def reset(self):
            obs = self._env.reset()
            self.dis_to_goal = np.linalg.norm(obs[0:2] - [1.0, 0.0])
            return obs

        def step(self, action):
            r = 0.0
            for _ in range(1):
                obs, reward, done, info = self._env.step(action)
                # if reward != 0.0:
                #     done = True
                # else:
                #     done = False
                if reward < 0.0:
                    reward = 0.0
                # reward -= 0.00175

                # if obs[0] < 0.0:
                #     done = True

                if not done:   # when env is done, ball position will be reset.
                    reward += self.incentive(obs)

                r += reward

                if done:
                    return obs, r * 100, done, info

            return obs, r*100, done, info

        def incentive(self, obs):
            # total accumulative incentive reward is around 0.5
            dis_to_goal_new = np.linalg.norm(obs[0:2] - [1.01, 0.0])
            r = 0.25*(self.dis_to_goal - dis_to_goal_new)
            self.dis_to_goal = dis_to_goal_new
            return r

        def incentive1(self, obs):
            r = -self.dis_to_goal*(1e-4)   # punishment weighted by dis_to_goal
            self.dis_to_goal = np.linalg.norm(obs[0:2] - [1.01, 0.0])  # interval: 0.0 ~ 2.0
            return r

        def incentive2(self, obs):
            who_controls_ball = obs[7:9]
            pos_ball = obs[0]
            distance_to_goal =np.array([(pos_ball+1)/2.0, (pos_ball-1)/2.0])
            r = np.dot(who_controls_ball,distance_to_goal)*0.003
            return r


    # academy_empty_goal academy_empty_goal_close
    env0 = football_env.create_environment(env_name=args.env, representation='simple115', render=False)
    env_1 = FootballWrapper(env0)
    env_3 = env_1


    # class Wrapper(object):
    #
    #     def __init__(self, env, action_repeat):
    #         self._env = env
    #         self.action_repeat = action_repeat
    #
    #     def __getattr__(self, name):
    #         return getattr(self._env, name)
    #
    #     def step(self, action):
    #         r = 0.0
    #         for _ in range(self.action_repeat):
    #             obs_, reward_, done_, info_ = self._env.step(action)
    #             reward_ = reward_ if reward_ > -99.0 else 0.0
    #             r = r + reward_
    #             if done_:
    #                 return obs_, r, done_, info_
    #         return obs_, r, done_, info_
    # env_1 = gym.make(args.env)
    # env_3 = Wrapper(gym.make(args.env),1)



    maxsqn(args, lambda n : env_3 if n==3 else env_1, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=args.net), replay_size=args.replay_size, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, start_steps=args.start_steps, save_freq=args.save_freq, Ln=args.Ln,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, alpha=args.alpha, lr=args.lr, max_ep_len = args.max_ep_len,
        logger_kwargs=logger_kwargs)