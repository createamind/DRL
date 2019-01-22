
import random
import numpy as np
import tensorflow as tf


# random.seed(3)
print(random.random())

# np.random.seed(4)
print(np.random.random())

# tf.set_random_seed(5)
print(tf.get_default_graph())


x = tf.Variable(0.0)
y = tf.Variable(0.0)
z = tf.Variable(0.0)
z_plus_1 = tf.assign_add(z, 1)
x_plus_1 = tf.assign_add(x, 1)
y_plus_1 = tf.assign_add(y, 1)
index = tf.random_uniform(minval=0, maxval=2, shape=[], dtype=tf.int32)

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(y)

with tf.control_dependencies([y_plus_1]):
    x = tf.identity(x)

with tf.control_dependencies([y_plus_1]):
    z = tf.identity(z)


e = tf.Variable(0.0)
d = tf.Variable(0.0)
b = tf.Variable(0.0)
with tf.control_dependencies([z_plus_1]):
    c=e+d
a=tf.add(c,b)
f=tf.Variable(0.0)

sess=tf.Session()
sess.run(tf.initialize_all_variables())


print(sess.run([a,f]))
print(sess.run(z))
print(sess.run(index))


for i in range(5):
    #print(sess.run(x),sess.run(y),sess.run(z))
    print(sess.run([a,x,y,z]))

# for j in range(ep_len):
#     for i in range(ensemble_size):
#         batch = replay_buffer.sample_batch(batch_size)
#         feed_dict = {x_ph: batch['obs1'],
#                      x2_ph: batch['obs2'],
#                      a_ph: batch['acts'],
#                      r_ph: batch['rews'],
#                      d_ph: batch['done'],
#                     }
#         # step_ops = [q1_loss, q1, logp_pi_, alpha, train_value_op, target_update, train_alpha_op]
#         sess.run(train_value_op[i])
#
#     outs = sess.run(step_ops, feed_dict)
#     logger.store(LossQ1=outs[0], Q1Vals=outs[1],
#                 LogPi=outs[2], Alpha=outs[3])