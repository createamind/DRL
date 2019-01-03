"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

code based on: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.3
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
import time, os, sys


EP_MAX = 1000
EP_LEN = 200

N_WORKER = 4  # parallel workers
N_WORKER_READY = 0

ACTION_LIMIT = 2.

GAMMA = 0.9  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 10  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 3, 1  # state and action dimension


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)  # V(s)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.c_loss = tf.reduce_mean(tf.square(self.advantage))
        self.c_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.c_loss)

        # actor
        pi, pi_params = self._build_a_net('pi', trainable=True)   # pi is a policy with a normal distribution N(mu,sigma), pi_params are the tf.GraphKeys.GLOBAL_VARIABLES
        oldpi, oldpi_params = self._build_a_net('oldpi', trainable=False)
        self.sample_op = tf.squeeze(oldpi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        #ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        #self.ratio = pi.prob(self.tfa) / ( oldpi.prob(self.tfa) )
        #self.tfratio = tf.placeholder
        self.tfoldpi_prob = tf.placeholder(tf.float32, [None, 1], 'oldpi_prob')
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        self.a_loss = -tf.reduce_mean(tf.minimum(ratio * self.tfadv,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.a_train_op = tf.train.AdamOptimizer(A_LR).minimize(self.a_loss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        #self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        #for _ in (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
        #    print(_)

        if not os.path.exists('save/'):
            os.makedirs('save/')

    def save(self,step):
        # timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        # self.saver.save(self.sess, sys.path[0]+'/save/ppo-{}'.format(timestamp), global_step=step, write_meta_graph=False, write_state=False)
        self.saver.save(self.sess, sys.path[0] + '/save/ppo-{}'.format(step), write_meta_graph=False, write_state=False)
        print('Trainable Variables Saved.')

    def restore(self,step):
        self.saver.restore(self.sess, sys.path[0]+'/save/ppo-{}'.format(step))
        print("Trainable Variables Loaded.")


    def update(self):
        global GLOBAL_UPDATE_COUNTER, N_WORKER_READY
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                #t0 = time.time()
                UPDATE_EVENT.wait()  # wait until get batch of data
                #t1 = time.time()
#                print(QUEUE.qsize())
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
#                print("data.shape:",s.shape,a.shape,r.shape)
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop

                #oldpi_pro = self.sess.run(self.oldpi.prob(self.tfa),{self.tfs:s, self.tfa:a})
#                print(oldpi_pro)
                [self.sess.run(self.a_train_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.c_train_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

                self.sess.run(self.update_oldpi_op)  # copy pi to old pi

                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available
#                print("++++++++++++++updated++++++++++++")
                N_WORKER_READY = 0
                #print(time.time()-t1,"-----",t1-t0)
                if GLOBAL_EP%100 == 0:
                    self.save(GLOBAL_EP)

    def _build_a_net(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = ACTION_LIMIT * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -ACTION_LIMIT, ACTION_LIMIT)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, N_WORKER_READY
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                #buffer_r.append(r)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
#                    print("===============QUEUE.put--- worker:", self.wid, "-------T:", br.shape)

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        N_WORKER_READY += 1
                        UPDATE_EVENT.set()
#                        if N_WORKER_READY >= N_WORKER:
#                            UPDATE_EVENT.set()  # globalPPO update
#                        print("===============UPDATE_EVENT.set()--- worker:", self.wid )

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
                #GLOBAL_RUNNING_R.append(ep_r )
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )


def main(args):

    global GLOBAL_PPO, UPDATE_EVENT, ROLLING_EVENT, GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_RUNNING_R, COORD, QUEUE

    GLOBAL_PPO = PPO()

    if not args.Test:
        # Training

        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
        UPDATE_EVENT.clear()  # not update now
        ROLLING_EVENT.set()  # start to roll out
        workers = [Worker(wid=i) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()  # workers putting data in this queue
        threads = []
        for worker in workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()  # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
        threads[-1].start()
        COORD.join(threads)

        # plot reward change and test
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel('Episode')
        plt.ylabel('Moving reward')
        #plt.ion()
        plt.show()

        env = gym.make('Pendulum-v0')
        while True:
            s = env.reset()
            s0 = env.env.state
            s00 = s
            totalreward = 0
            for t in range(200):
                #env.render()
                s, r_test, _, _ = env.step(GLOBAL_PPO.choose_action(s))
                totalreward += r_test
            print("reward: ",totalreward)

            if totalreward <= -550:
                print("Bad Result Alert")
                env.reset()
                env.env.state = s0
                print(env.env.state)
                totalreward1 = 0
                for t in range(200):
                    env.render()
                    s00,r_test1,_,_ = env.step(GLOBAL_PPO.choose_action(s00))
                    totalreward1 += r_test1
                print("reward: ", totalreward1)
                print("===============================================")

    else:
        #Testing

        GLOBAL_PPO.restore(900)
        factor1 = 2*np.pi/1000
        vtest = GLOBAL_PPO.sess.run(GLOBAL_PPO.v, {GLOBAL_PPO.tfs: [[np.cos(th*factor1),np.sin(th*factor1),0] for th in range(1000)]})
        #print(vtest)
        plt.plot(np.arange(len(vtest))*factor1, vtest)
        plt.xlabel('theta')
        plt.ylabel('value')
        #plt.ion()
        plt.show()


        env = gym.make('Pendulum-v0')

        env.reset()
        state = [np.pi,0]
        s = np.array([np.cos(state[0]),np.sin(state[0]),state[1]])
        env.env.state = state
        totalreward = 0
        for t in range(200):
            env.render()
            s, r_test, _, _ = env.step(GLOBAL_PPO.choose_action(s))
            #totalreward += ((r_test+8)/8)*0.9**t
            totalreward += r_test
        print("reward: ",totalreward)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Test', type=bool, default=True)
    args = parser.parse_args()

    main(args)