import numpy as np
import tensorflow as tf
import tflearn
GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.005
ENTROPY_EPS = 1e-06
S_INFO = 4
S_LEN = 8
DEFAULT_QUALITY = 1
VIDEO_BIT_RATE = [300,
                  750,
                  1200,
                  1850,
                  2850,
                  4300]
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
CLIP_USING_NORM = True
CLIP_NORM_MAGNITUDE = 40.0
CLIP_VALUE_MAGNITUDE = 0.001


def modify_gradients(x):
    x = [tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad) for grad in x]
    x = [tf.where(tf.equal(grad, np.inf), tf.zeros_like(grad), grad)
         for grad in x]
    x = [tf.where(tf.equal(grad, -np.inf), tf.zeros_like(grad), grad)
         for grad in x]
    if CLIP_USING_NORM == True:
        x, _ = tf.clip_by_global_norm(x, CLIP_NORM_MAGNITUDE)
    else:
        x = [tf.clip_by_value(grad, -CLIP_VALUE_MAGNITUDE, +
                              CLIP_VALUE_MAGNITUDE) for grad in x]
    return x


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """

    def __init__(self, state_dim, action_dim, learning_rate, global_workers, scope):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.trainer = tf.train.AdamOptimizer
        self.scope = scope
        self.inputs, self.out = self.create_actor_network(scope)
        if 'global' in scope:
            self.block = tf.Variable(
                initial_value=[True], dtype=tf.bool, trainable=False, name='blocker_actor')
        if 'global' not in scope:
            self.update_local_ops = update_target_graph(
                'actor_global_' + str(scope[-1]), scope)
            self.transfer_global = [update_target_graph(
                'actor_global_0', 'actor_global_' + str(i)) for i in range(global_workers)]
            self.network_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
            self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])
            self.obj = tf.reduce_sum(tf.multiply(tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts), reduction_indices=1, keep_dims=True)), -
                                     self.act_grad_weights)) + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out, tf.log(self.out + ENTROPY_EPS)))
            self.actor_gradients = tf.gradients(self.obj, self.network_params)
            self.optimize = self.trainer(self.lr_rate).apply_gradients(
                zip(self.actor_gradients, self.network_params))
            self.own_global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_global_' + str(scope[-1]))
            self.apply_own_grads = self.trainer(self.lr_rate).apply_gradients(
                zip(self.actor_gradients, self.own_global_vars))
            self.others_list = [int(x) for x in range(
                global_workers) if x != int(str(scope[-1]))]
            other_global_vars = [tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_global_' + str(i)) for i in self.others_list]
            self.feed_gradients = [tf.reshape(tf.placeholder(
                shape=[None], dtype=tf.float32), grad.shape) for grad in self.actor_gradients]
            self.apply_other_grads = [self.trainer(self.lr_rate).apply_gradients(zip(
                self.feed_gradients, other_global_vars[i])) for i in range(len(self.others_list))]
        return

    def create_actor_network(self, scope):
        with tf.variable_scope(scope):
            inputs = tflearn.input_data(
                shape=[None, self.s_dim[0], self.s_dim[1]])
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], 128, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], 128, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], 128, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 4:5, -1], 128, activation='relu')
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            merge_net = tflearn.merge([split_0,
                                       split_1,
                                       split_2_flat,
                                       split_3_flat,
                                       split_4_flat,
                                       split_5], 'concat')
            dense_net_0 = tflearn.fully_connected(
                merge_net, 128, activation='relu')
            out = tflearn.fully_connected(
                dense_net_0, self.a_dim, activation='softmax')
            return (inputs, out)
        return

    def train(self, sess, inputs, acts, act_grad_weights):
        sess.run(self.optimize, feed_dict={self.inputs: inputs,
                                           self.acts: acts,
                                           self.act_grad_weights: act_grad_weights})

    def predict(self, sess, inputs):
        return sess.run(self.out, feed_dict={self.inputs: inputs})

    def get_gradients(self, sess, inputs, acts, act_grad_weights):
        return sess.run(self.actor_gradients, feed_dict={self.inputs: inputs,
                                                         self.acts: acts,
                                                         self.act_grad_weights: act_grad_weights})

    def apply_gradients(self, sess, actor_gradients):
        return sess.run([self.optimize, self.apply_own_grads], feed_dict={i: d for i, d in zip(self.actor_gradients, actor_gradients)})

    def get_network_params(self, sess):
        return sess.run(self.network_params)

    def set_network_params(self, sess, input_network_params):
        sess.run(self.set_network_params_op, feed_dict={
                 i: d for i, d in zip(self.input_network_params, input_network_params)})

    def get_block_vars(self, sess):
        return sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'blocker_actor' in x.name])

    def transfer_global_params(self, sess):
        sess.run(self.transfer_global)

    def update_local_params(self, sess):
        sess.run(self.update_local_ops)


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """

    def __init__(self, state_dim, learning_rate, global_workers, scope):
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.trainer = tf.train.AdamOptimizer
        self.scope = scope
        self.inputs, self.out = self.create_critic_network(scope)
        if 'global' in scope:
            self.block = tf.Variable(
                initial_value=[True], dtype=tf.bool, trainable=False, name='blocker_critic')
        if 'global' not in scope:
            self.update_local_ops = update_target_graph(
                'critic_global_' + str(scope[-1]), scope)
            self.transfer_global = [update_target_graph(
                'critic_global_0', 'critic_global_' + str(i)) for i in range(global_workers)]
            self.network_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.td_target = tf.placeholder(tf.float32, [None, 1])
            self.td = tf.subtract(self.td_target, self.out)
            self.loss = tflearn.mean_square(self.td_target, self.out)
            self.critic_gradients = tf.gradients(
                self.loss, self.network_params)
            self.optimize = self.trainer(self.lr_rate).apply_gradients(
                zip(self.critic_gradients, self.network_params))
            self.own_global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_global_' + str(scope[-1]))
            self.apply_own_grads = self.trainer(self.lr_rate).apply_gradients(
                zip(self.critic_gradients, self.own_global_vars))
            self.others_list = [int(x) for x in range(
                global_workers) if x != int(str(scope[-1]))]
            other_global_vars = [tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'critic_global_' + str(i)) for i in self.others_list]
            self.feed_gradients = [tf.reshape(tf.placeholder(
                shape=[None], dtype=tf.float32), grad.shape) for grad in self.critic_gradients]
            self.apply_other_grads = [self.trainer(self.lr_rate).apply_gradients(zip(
                self.feed_gradients, other_global_vars[i])) for i in range(len(self.others_list))]
        return

    def create_critic_network(self, scope):
        with tf.variable_scope(scope):
            inputs = tflearn.input_data(
                shape=[None, self.s_dim[0], self.s_dim[1]])
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], 128, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], 128, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], 128, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 4:5, -1], 128, activation='relu')
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            merge_net = tflearn.merge([split_0,
                                       split_1,
                                       split_2_flat,
                                       split_3_flat,
                                       split_4_flat,
                                       split_5], 'concat')
            dense_net_0 = tflearn.fully_connected(
                merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')
            return (inputs, out)
        return

    def train(self, sess, inputs, td_target):
        return sess.run([self.loss, self.optimize], feed_dict={self.inputs: inputs,
                                                               self.td_target: td_target})

    def predict(self, sess, inputs):
        return sess.run(self.out, feed_dict={self.inputs: inputs})

    def get_td(self, sess, inputs, td_target):
        return sess.run(self.td, feed_dict={self.inputs: inputs,
                                            self.td_target: td_target})

    def get_gradients(self, sess, inputs, td_target):
        return sess.run(self.critic_gradients, feed_dict={self.inputs: inputs,
                                                          self.td_target: td_target})

    def apply_gradients(self, sess, critic_gradients):
        return sess.run([self.optimize, self.apply_own_grads], feed_dict={i: d for i, d in zip(self.critic_gradients, critic_gradients)})

    def get_network_params(self, sess):
        return sess.run(self.network_params)

    def set_network_params(self, sess, input_network_params):
        sess.run(self.set_network_params_op, feed_dict={
                 i: d for i, d in zip(self.input_network_params, input_network_params)})

    def get_block_vars(self, sess):
        return sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'blocker_critic' in x.name])

    def update_global_params(self, sess, grads, other_ids):
        num_params = len(other_ids)
        feed_dict = {k: v for k, v in zip(self.feed_gradients, grads)}
        for i in range(num_params + 1):
            sess.run(self.block_global[int(i)])

        for i in range(num_params):
            sess.run(
                self.local_AC.apply_other_grads[int(i)], feed_dict=feed_dict)

        for i in range(num_params + 1):
            sess.run(self.unblock_global[int(i)])

    def transfer_global_params(self, sess):
        sess.run(self.transfer_global)

    def update_local_params(self, sess):
        sess.run(self.update_local_ops)


def update_target_graph(from_scope, to_scope):
    """Transfers weights from one scope to another"""
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


def compute_gradients(sess, s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """

    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]

    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch
    actor_gradients = actor.get_gradients(sess, s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(sess, s_batch, R_batch)
    return (actor_gradients, critic_gradients, td_batch)


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]

    raise x.ndim >= 1 or AssertionError
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])

    return H


def build_summaries():
    td_loss = tf.Variable(0.0)
    tf.summary.scalar('TD_loss', td_loss)
    eps_total_reward = tf.Variable(0.0)
    tf.summary.scalar('Eps_total_reward', eps_total_reward)
    avg_entropy = tf.Variable(0.0)
    tf.summary.scalar('Avg_entropy', avg_entropy)
    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()
    return (summary_ops, summary_vars)
