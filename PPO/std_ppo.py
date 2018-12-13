import numpy as np
import tensorflow as tf

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class PPO(object):
    def __init__(self,
                 sess,
                 state_dim,
                 action_dim,
                 action_bound,
                 c1,  # value loss coefficient
                 c2,  # entropy coefficient
                 epsilon,  # clip epsilon
                 lr,
                 K):  # train K epochs
        self.sess = sess

        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_bound = action_bound
        self.K = K

        self.pl_s = tf.placeholder(tf.float32, shape=(None, state_dim), name='s_t')

        pi, self.v, params = self._build_net(self.pl_s, 'policy', True)
        old_pi, old_v, old_params = self._build_net(self.pl_s, 'old_policy', False)

        self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_r')

        advantage = self.pl_discounted_r - old_v

        self.pl_a = tf.placeholder(tf.float32, shape=(None, action_dim), name='a_t')
        ratio = pi.prob(self.pl_a) / old_pi.prob(self.pl_a)

        L_clip = tf.math.reduce_mean(tf.math.minimum(
            ratio * advantage,  # surrogate objective
            tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
        ))
        L_vf = tf.reduce_mean(tf.square(self.pl_discounted_r - self.v))
        S = tf.reduce_mean(pi.entropy())
        L = L_clip - c1 * L_vf + c2 * S

        self.choose_action_op = tf.squeeze(pi.sample(1), axis=0)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-L)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]

    def _build_net(self, inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(inputs, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)

            prob_l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(prob_l, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(prob_l, self.a_dim, tf.nn.softplus, trainable=trainable, **initializer_helper)

            v_l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v_l = tf.layers.dense(v_l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(v_l, 1, trainable=trainable, **initializer_helper)

            mu, sigma = mu * self.a_bound, sigma
            if trainable:
                self.mu = mu
                self.sigma = sigma

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

            params = tf.get_variable_scope().global_variables()
            
        return norm_dist, v, params

    def get_v(self, s):
        assert len(s.shape) == 1

        return self.sess.run(self.v, {
            self.pl_s: np.array(s[np.newaxis, :])
        }).squeeze()

    def test(self, s):
        mu, sigma = self.sess.run([self.mu, self.sigma], {
            self.pl_s: s
        })
        v = self.sess.run(self.v, {
            self.pl_s: s
        })
        a = self.sess.run(self.choose_action_op, {
            self.pl_s: s
        })
        for i in range(len(mu)):
            print(mu[i], sigma[i], v[i], a[i])

    def choose_action(self, s):
        assert len(s.shape) == 2

        a = self.sess.run(self.choose_action_op, {
            self.pl_s: s
        })
        return np.clip(a, -self.a_bound, self.a_bound)

    def train(self, s, a, discounted_r):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(discounted_r.shape) == 2

        self.sess.run(self.update_params_op)

        # K epochs
        for i in range(self.K):
            self.sess.run(self.train_op, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_discounted_r: discounted_r
            })
