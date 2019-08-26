
import tensorflow as tf
import gym
import joblib
import os
import os.path as osp
from spinup.utils.logx import restore_tf_graph


def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BipedalWalkerHardcore-v2')
    parser.add_argument('--render', type=bool, default=True)
    args = parser.parse_args()

    file_model = '/home/liu/project/DRL/saved_models/biped_sac1_stump6_actnoise0.3_alphaauto_2/biped_sac1_stump6_actnoise0.3_alphaauto_2_s0'
    print(file_model)

    _env, get_action = load_policy(file_model, deterministic=True)

    test_env = gym.make(args.env)
    ave_ep_ret = 0
    for j in range(10000):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not d: # (d or (ep_len == 2000)):
            o, r, d, _ = test_env.step(get_action(o))
            ep_ret += r
            ep_len += 1
            if args.render:
                test_env.render()
        ave_ep_ret = (j*ave_ep_ret + ep_ret)/(j+1)
        print('ep_len', ep_len, 'ep_ret:', ep_ret, 'ave_ep_ret:',ave_ep_ret,'({}/10000)'.format(j+1) )
    


