import numpy as np
import tensorflow as tf


num = int(1e7)

logits = np.array([1.0, 2.3])[np.newaxis,...]

logits = tf.nn.log_softmax(logits, axis=1)

rands = tf.random.multinomial(logits, num)

sess=tf.Session()
sess.run(tf.initialize_all_variables())


rands1 = sess.run(rands)
print(np.sum(rands1)/num)


# 0.7858171
# 0.7858977