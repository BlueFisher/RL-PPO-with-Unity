import time

import numpy as np
import tensorflow as tf

from util.saver import Saver

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 saver_model_path='model',
                 summary_path='log',
                 summary_name=None,
                 write_summary_graph=False,
                 variance_bound=2.,
                 batch_size=2048,
                 c1=1,
                 c2=0.001,  # entropy coefficient
                 epsilon=0.2,  # clip epsilon
                 init_lr=0.00005,
                 epoch_size=10):  # train K epochs

        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                               graph=self.graph)

        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_bound = action_bound
        self.variance_bound = variance_bound
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        with self.graph.as_default():
            self._build_model(c1, c2, epsilon, init_lr)
            self.saver = Saver(saver_model_path, self.sess)
            self.init_iteration = self.saver.restore_or_init()

            self.summary_writer = None
            if summary_path is not None:
                if write_summary_graph:
                    writer = tf.summary.FileWriter(summary_path, self.graph)
                    writer.close()

                if summary_name is None:
                    summary_name = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
                self.summary_writer = tf.summary.FileWriter(f'{summary_path}/{summary_name}')

        self._last_cache_time = time.time()
        self._cache_variables_cachable()

    def _build_model(self, c1, c2, epsilon, init_lr):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='s_t')

        policy, policy_variables = self._build_net(self.pl_s, 'policy', True)
        old_policy, old_policy_variables = self._build_net(self.pl_s, 'old_policy', False)

        self.v, v_variables = self._build_critic_net(self.pl_s, 'value', True)
        old_v, old_v_variables = self._build_critic_net(self.pl_s, 'old_value', False)

        self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_r')

        advantage = self.pl_discounted_r - old_v

        self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='a_t')
        ratio = policy.prob(self.pl_a) / old_policy.prob(self.pl_a)

        L_clip = tf.math.reduce_mean(tf.math.minimum(
            ratio * advantage,  # surrogate objective
            tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
        ))
        L_vf = tf.reduce_mean(tf.square(self.pl_discounted_r - self.v))
        S = tf.reduce_mean(policy.entropy())
        L = L_clip - c1 * L_vf + c2 * S

        self.choose_action_op = tf.squeeze(policy.sample(1), axis=0)

        self.lr = tf.get_variable('lr', shape=(), initializer=tf.constant_initializer(init_lr))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-L)

        self.update_variables_op = [tf.assign(r, v) for r, v in
                                    zip(old_policy_variables + old_v_variables,
                                        policy_variables + v_variables)]

        self.variables_cachable = [v for v in tf.global_variables() if v != self.lr]
        self.variables_cached_tmp = self.variables_cached = None

        tf.summary.scalar('loss/-clipped_objective', L_clip)
        tf.summary.scalar('loss/value_function', L_vf)
        tf.summary.scalar('loss/-entropy', S)
        tf.summary.scalar('loss/-mixed_objective', L)
        tf.summary.scalar('loss/lr', self.lr)
        self.summaries = tf.summary.merge_all()

    def _build_critic_net(self, s_inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(l, 1, trainable=trainable)

            variables = tf.get_variable_scope().global_variables()

        return v, variables

    def _build_net(self, s_inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)

            mu = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.softplus, trainable=trainable, **initializer_helper)

            mu, sigma = mu * self.a_bound, tf.clip_by_value(sigma, 0, self.variance_bound)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

            variables = tf.get_variable_scope().global_variables()

        return norm_dist, variables

    def get_v(self, s):
        assert len(s.shape) == 1

        return self.sess.run(self.v, {
            self.pl_s: np.array(s[np.newaxis, :])
        }).squeeze()

    def choose_action(self, s):
        assert len(s.shape) == 2

        a = self.sess.run(self.choose_action_op, {
            self.pl_s: s
        })

        # for i, ai in enumerate(a):
        #     for j in range(2):
        #         width = self.a_bound[j] * 2
        #         if self.a_bound[j] < ai[j] < self.a_bound[j] + width:
        #             ai[j] -= width
        #         elif -self.a_bound[j] - width < ai[j] < -self.a_bound[j]:
        #             ai[j] += width
        #         elif ai[j] <= -self.a_bound[j] - width or ai[j] >= self.a_bound[j] + width:
        #             ai[j] = np.random.rand(1) * 2 - 1
        #     a[i] = ai

        if np.isnan(np.min(a)):
            print('WARNING! NAN IN ACTIONS')
            self._restore_variables_cachable()
            self._decrease_lr()
            a = self.sess.run(self.choose_action_op, {
                self.pl_s: s
            })

        return np.clip(a, -self.a_bound, self.a_bound)

    def _decrease_lr(self, delta=2):
        lr = self.sess.run(self.lr)
        return self.sess.run(self.lr.assign(lr / delta))

    def save_model(self, iteration):
        self.saver.save(iteration)

    def _restore_variables_cachable(self):
        self.sess.run([tf.assign(r, v) for r, v in
                       zip(self.variables_cachable, self.variables_cached)])

    def _cache_variables_cachable(self):
        self.variables_cached = self.variables_cached_tmp
        self.variables_cached_tmp = self.sess.run(self.variables_cachable)
        if self.variables_cached is None:
            self.variables_cached = self.variables_cached_tmp

    def write_constant_summaries(self, constant_summaries, iteration):
        if self.summary_writer is not None:
            summaries = tf.Summary(value=[tf.Summary.Value(tag=i['tag'],
                                                           simple_value=i['simple_value'])
                                          for i in constant_summaries])
            self.summary_writer.add_summary(summaries, iteration)

    def train(self, s, a, discounted_r, iteration):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(discounted_r.shape) == 2
        assert s.shape[0] == a.shape[0] == discounted_r.shape[0]

        self.sess.run(self.update_variables_op)

        now = time.time()
        if now - self._last_cache_time >= 600:
            self._cache_variables_cachable()
            self._last_cache_time = now  # UPDATE

        if self.summary_writer is not None:
            summaries = self.sess.run(self.summaries, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_discounted_r: discounted_r
            })
            self.summary_writer.add_summary(summaries, iteration)

        for i in range(0, s.shape[0], self.batch_size):
            _s, _a, _discounted_r = (s[i:i + self.batch_size],
                                     a[i:i + self.batch_size],
                                     discounted_r[i:i + self.batch_size])
            for _ in range(self.epoch_size):
                self.sess.run(self.train_op, {
                    self.pl_s: _s,
                    self.pl_a: _a,
                    self.pl_discounted_r: _discounted_r
                })
