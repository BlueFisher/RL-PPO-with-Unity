import gym
import numpy as np
import tensorflow as tf
from memory import Memory


initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.3),
    'bias_initializer': tf.constant_initializer(0.1)
}


class DQN(object):
    def __init__(self,
                 sess,
                 state_dim,  # state dimension
                 action_dim,  # length of one hot actions
                 batch_size,
                 memory_max_size,
                 gamma,
                 lr,  # learning rate
                 epsilon,  # epsilon-greedy
                 replace_target_iter):  # frequency of updating target params
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter

        self.memory = Memory(batch_size, memory_max_size)
        self._learn_step_counter = 0
        self._generate_model()

    def choose_action(self, s, use_epsilon):
        assert len(s.shape) == 1
        if use_epsilon:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.a_dim)
            else:
                qs = self.sess.run(self.qs_eval, feed_dict={
                    self.pl_s: s[np.newaxis, :]
                })
                return int(qs.squeeze().argmax())
        else:
            qs = self.sess.run(self.qs_eval, feed_dict={
                self.pl_s: s[np.newaxis, :]
            })
            return int(qs.squeeze().argmax())

    def _generate_model(self):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s')
        self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a')
        self.pl_r = tf.placeholder(tf.float32, shape=(None, 1), name='r')
        self.pl_s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_')
        self.pl_done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        self.qs_eval, param_eval = self._build_net(self.pl_s, 'eval_net', True)
        qs_target, param_target = self._build_net(self.pl_s_, 'target_net', False)

        # argmax(Q)
        max_a = tf.argmax(self.qs_eval, axis=1)
        one_hot_max_a = tf.one_hot(max_a, self.a_dim)

        # y = R + gamma * Q_(S, argmax(Q))
        q_target = self.pl_r + self.gamma \
            * tf.reduce_sum(one_hot_max_a * qs_target, axis=1, keepdims=True) * (1 - self.pl_done)
        q_target = tf.stop_gradient(q_target)

        q_eval = tf.reduce_sum(self.pl_a * self.qs_eval, axis=1, keepdims=True)

        self.loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # replace target network params with eval network params
        self.target_replace_ops = [tf.assign(t, e) for t, e in zip(param_target, param_eval)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            qs = tf.layers.dense(l, self.a_dim, trainable=trainable, **initializer_helper)

        return qs, tf.global_variables(scope=scope)

    def test(self, s):
        print(self.sess.run(self.qs_eval, {
            self.pl_s: s
        }))

    def store_transition_and_learn(self, s, a_i, r, s_, done):
        assert len(s.shape) == 1
        assert type(a_i) == int
        assert type(r) == list
        assert len(s_.shape) == 1
        assert type(done) == list

        if self._learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_ops)

        # turn action to one hot form
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a_i] = 1

        self.memory.store_transition(s, one_hot_action, r, s_, done)
        if self.memory.isFull:
            self._learn()

        self._learn_step_counter += 1

    def _learn(self):
        s, a, r, s_, done = self.memory.get_mini_batches()

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_s_: s_,
            self.pl_done: done
        })
