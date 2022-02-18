import tensorflow as tf
from a3c_gs import ActorNetwork
import pickle
import sys
actor = ActorNetwork(state_dim=[6, 8], action_dim=6, learning_rate=None,
                     global_workers=None, entropy_weight=None, scope='global_actor')

from_var = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='global_actor')
to_var = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='global_actor')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    froom = sess.run(from_var)

    size = 0

    for x in froom:
        size += x.size * x.itemsize

    print(size)

    # print(sys.getsizeof(froom))
    # f = open('yeet.pickle', 'wb')
    # pickle.dump(froom, f)
    # f.close()

    # f = open('yeet.pickle', 'rb')
    # froom = pickle.load(f)

    # from_var[0].load(froom[0], sess)
    # to = sess.run(to_var)
