import numpy as np
import tensorflow as tf

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class Actor(object):
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 action_bound,
                 lr,
                 tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_bound = action_bound
        self.lr = lr

        with tf.variable_scope('actor'):
            self.pl_s = tf.placeholder(tf.float32, shape=(None, state_dim), name='state')
            self.pl_s = tf.identity(self.pl_s, name='vector_observations')
            self.pl_s_ = tf.placeholder(tf.float32, shape=(None, state_dim), name='state_')

            self.a, self.param_eval = self._build_net(self.pl_s, 'eval', True)
            self.a = tf.identity(self.a, name='action')
            self.a_, param_target = self._build_net(self.pl_s_, 'target', False)

        # soft update
        self.target_replace_op = [tf.assign(t, tau * e + (1 - tau) * t)
                                  for t, e in zip(param_target, self.param_eval)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(
                s, 32, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )
            l = tf.layers.dense(
                l, 32, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )
            l = tf.layers.dense(
                l, 32, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )

            a = tf.layers.dense(
                l, self.a_dim, activation=tf.nn.tanh,
                trainable=trainable, **initializer_helper
            )
            a = a * self.a_bound

            params = tf.get_variable_scope().global_variables()

        return a, params

    def choose_action(self, s, variance=None):
        assert len(s.shape) == 1
        a = self.sess.run(self.a, {
            self.pl_s: s[np.newaxis, :]
        })

        assert len(a.shape) == 2

        action = a[0]
        if variance is not None:
            action = np.clip(np.random.normal(action, variance), -self.a_bound, self.a_bound)  # exploration

        return action

    def generate_gradients(self, Q_a_gradients):
        #  generate actor's gradient according to chain rule
        grads = tf.gradients(self.a, self.param_eval, Q_a_gradients)
        optimizer = tf.train.AdamOptimizer(-self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, self.param_eval))

    def learn(self, s):
        self.sess.run(self.train_op, {
            self.pl_s: s
        })
        self.sess.run(self.target_replace_op)


class Critic(object):
    def __init__(self, sess, pl_s, pl_s_, a, a_, gamma, lr, tau):
        self.sess = sess
        self.pl_s = pl_s
        self.pl_s_ = pl_s_
        self.pl_a = a

        with tf.variable_scope('critic'):
            self.pl_r = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
            self.pl_done = tf.placeholder(tf.float32, shape=(None, 1), name='dones')
            self.q, param_eval = self._build_net(pl_s, a, 'eval', True)
            self.q_, param_target = self._build_net(pl_s_, a_, 'target', False)

        # soft update
        self.target_replace_op = [tf.assign(t, tau * e + (1 - tau) * t)
                                  for t, e in zip(param_target, param_eval)]

        # y_t
        target_q = self.pl_r + gamma * self.q_
        # reserve or ignore the gradient of target_q
        target_q = tf.stop_gradient(target_q)

        loss = tf.reduce_mean(tf.squared_difference(target_q, self.q))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=param_eval)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            ls = tf.layers.dense(
                s, 32, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )
            la = tf.layers.dense(
                a, 32, activation=tf.nn.relu,
                trainable=trainable, **initializer_helper
            )
            # l = ls + la
            l = tf.concat([ls, la], 1)
            l = tf.layers.dense(l, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, activation=tf.nn.relu, trainable=trainable, **initializer_helper)

            q = tf.layers.dense(l, 1, name='q', trainable=trainable, **initializer_helper)

            params = tf.get_variable_scope().global_variables()
        return q, params

    # generate the derivative of Q with respect to a and transfer to actor
    def get_gradients(self):
        return tf.gradients(self.q, self.pl_a)[0]

    def learn(self, s, a, r, s_, done):
        self.sess.run(self.train_op, {
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_s_: s_,
            self.pl_done: done
        })
        self.sess.run(self.target_replace_op)
