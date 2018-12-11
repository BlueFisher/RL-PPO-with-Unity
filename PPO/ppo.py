import numpy as np
import tensorflow as tf

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class PPO(object):
    def __init__(self, sess, s_dim, a_dim, a_bound, c1, c2, epsilon, lr, K):
        self.sess = sess

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.K = K

        self.pl_s = tf.placeholder(tf.float32, shape=(None, s_dim), name='s_t')
        self.pl_sigma = tf.placeholder(tf.float32, shape=(), name='sigma')

        pi, params = self._build_net(self.pl_s, 'policy', True)
        old_pi, old_params = self._build_net(self.pl_s, 'old_policy', False)

        self.v, v_params = self._build_net_c(self.pl_s, 'value', True)
        old_v, old_v_params = self._build_net_c(self.pl_s, 'old_value', False)

        self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_r')

        advantage = self.pl_discounted_r - old_v

        self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
        ratio = pi.prob(self.pl_a) / old_pi.prob(self.pl_a)

        L_clip = tf.math.reduce_mean(tf.math.minimum(
            ratio * advantage,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
        ))
        L_vf = tf.reduce_mean(tf.square(self.pl_discounted_r - self.v))
        S = tf.reduce_mean(pi.entropy())
        L = L_clip + c2 * S

        self.choose_action_op = tf.squeeze(pi.sample(1), axis=0)
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(-L)
        self.train_v_op = tf.train.AdamOptimizer(0.0001).minimize(L_vf)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]
        self.update_v_params_op = [tf.assign(r, v) for r, v in zip(old_v_params, v_params)]

    def _build_net_c(self, inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(inputs, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(l, 1, trainable=trainable)

            params = tf.global_variables(scope)
            return v, params

    def _build_net(self, inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(inputs, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)

            prob_l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(prob_l, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(prob_l, self.a_dim, tf.nn.softplus, trainable=trainable, **initializer_helper)

            # 状态价值函数 v 与策略 π 共享同一套神经网络参数
            # v_l = tf.layers.dense(l, 10, tf.nn.relu, trainable=trainable, **initializer_helper)
            # v = tf.layers.dense(v_l, 1, trainable=trainable, **initializer_helper)

            mu, sigma = mu * self.a_bound, sigma
            if trainable:
                self.mu = mu
                self.sigma = sigma

            norm_dist = tf.distributions.Normal(loc=mu, scale=self.pl_sigma)

        params = tf.global_variables(scope)
        return norm_dist, params

    def get_v(self, s):
        assert len(s.shape) == 1

        return self.sess.run(self.v, {
            self.pl_s: np.array(s[np.newaxis, :])
        }).squeeze()

    def test(self, s, sigma):
        mu = self.sess.run(self.mu, {
            self.pl_s: s
        })
        v = self.sess.run(self.v, {
            self.pl_s: s
        })
        a = self.sess.run(self.choose_action_op, {
            self.pl_s: s,
            self.pl_sigma: sigma
        })
        for i in range(len(mu)):
            print(mu[i], v[i], a[i])

    def choose_action(self, s, sigma):
        assert len(s.shape) == 2

        a = self.sess.run(self.choose_action_op, {
            self.pl_s: s,
            self.pl_sigma: sigma
        })
        return np.clip(a, -self.a_bound, self.a_bound)

    def train(self, s, a, discounted_r, sigma):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(discounted_r.shape) == 2

        self.sess.run(self.update_params_op)
        self.sess.run(self.update_v_params_op)

        # K epochs
        for i in range(self.K):
            self.sess.run(self.train_op, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_discounted_r: discounted_r,
                self.pl_sigma: sigma
            })
            self.sess.run(self.train_v_op, {
                self.pl_s: s,
                self.pl_discounted_r: discounted_r
            })
