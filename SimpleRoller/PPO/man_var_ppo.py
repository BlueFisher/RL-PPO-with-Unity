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
                 epsilon,  # clip epsilon
                 lr,
                 K):  # train K epochs
        self.sess = sess

        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_bound = action_bound
        self.K = K

        self.pl_s = tf.placeholder(tf.float32, shape=(None, state_dim), name='s_t')
        # constant variance
        self.sigma = tf.get_variable('sigma', shape=(),
                                     initializer=tf.initializers.constant(3.),
                                     trainable=False)

        pi, params = self._build_net(self.pl_s, 'policy', True)
        old_pi, old_params = self._build_net(self.pl_s, 'old_policy', False)

        self.v, v_params = self._build_critic_net(self.pl_s, 'value', True)
        old_v, old_v_params = self._build_critic_net(self.pl_s, 'old_value', False)

        self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_r')

        advantage = self.pl_discounted_r - old_v

        self.pl_a = tf.placeholder(tf.float32, shape=(None, action_dim), name='a_t')
        ratio = pi.prob(self.pl_a) / old_pi.prob(self.pl_a)

        L_clip = tf.math.reduce_mean(tf.math.minimum(
            ratio * advantage,  # surrogate objective
            tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
        ))
        L_vf = tf.reduce_mean(tf.square(self.pl_discounted_r - self.v))
        L = L_clip

        tf.summary.scalar('loss/-clipped_objective', L_clip)
        tf.summary.scalar('loss/value_function', L_vf)
        tf.summary.scalar('loss/-mixed_objective', L)
        tf.summary.scalar('variance', self.sigma)
        self.summaries = tf.summary.merge_all()

        self.choose_action_op = tf.squeeze(pi.sample(1), axis=0)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-L)
        self.train_v_op = tf.train.AdamOptimizer(lr).minimize(L_vf)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]
        self.update_v_params_op = [tf.assign(r, v) for r, v in zip(old_v_params, v_params)]

    def _build_critic_net(self, inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(inputs, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(l, 1, trainable=trainable)

            params = tf.get_variable_scope().global_variables()

        return v, params

    def _build_net(self, inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(inputs, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)

            prob_l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(prob_l, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)

            mu = mu * self.a_bound
            if trainable:
                self.mu = mu

            norm_dist = tf.distributions.Normal(loc=mu, scale=self.sigma)

            params = tf.get_variable_scope().global_variables()

        return norm_dist, params

    def get_v(self, s):
        assert len(s.shape) == 1

        return self.sess.run(self.v, {
            self.pl_s: np.array(s[np.newaxis, :])
        }).squeeze()

    def print_test(self, s):
        np.random.shuffle(s)
        mu, a = self.sess.run([self.mu, self.choose_action_op], {
            self.pl_s: s[:4]
        })

        for i in range(len(mu)):
            print(mu[i], a[i])

    def choose_action(self, s, sigma=None):
        assert len(s.shape) == 2

        if sigma is None:
            a = self.sess.run(self.choose_action_op, {
                self.pl_s: s
            })
        else:
            a = self.sess.run(self.choose_action_op, {
                self.pl_s: s,
                self.sigma: sigma
            })
        return np.clip(a, -self.a_bound, self.a_bound)

    def get_summaries(self, s, a, discounted_r):
        summaries = self.sess.run(self.summaries, {
            self.pl_s: s,
            self.pl_a: a,
            self.pl_discounted_r: discounted_r
        })

        return summaries

    def decrease_sigma(self):
        self.sess.run(self.sigma.assign(self.sigma * 0.999))

    def train(self, s, a, discounted_r):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(discounted_r.shape) == 2

        self.sess.run([self.update_params_op, self.update_v_params_op])

        # K epochs
        for i in range(self.K):
            self.sess.run([self.train_op, self.train_v_op], {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_discounted_r: discounted_r
            })
