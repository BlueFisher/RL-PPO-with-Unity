import numpy as np
import tensorflow as tf

from .ppo_base import PPO_Base

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class PG(PPO_Base):
    def _build_model(self, c1, c2, epsilon, init_lr):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state')

        policy = self._build_actor_net(self.pl_s, 'actor')
        self.v = self._build_critic_net(self.pl_s, 'critic')

        self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_reward')

        advantage = self.pl_discounted_r - self.v

        self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='action')

        L_vf = tf.math.reduce_mean(tf.square(advantage))
        L = tf.math.reduce_mean(policy.log_prob(self.pl_a) * advantage)

        self.choose_action_op = tf.squeeze(policy.sample(1), axis=0)

        with tf.name_scope('optimizer'):
            self.lr = tf.get_variable('lr', shape=(), initializer=tf.constant_initializer(init_lr))
            self.train_op = [tf.train.AdamOptimizer(self.lr).minimize(-L),
                             tf.train.AdamOptimizer(self.lr).minimize(L_vf)]

        self.variables_cachable = [v for v in tf.global_variables() if v != self.lr]

        tf.summary.scalar('loss/value_function', L_vf)
        tf.summary.scalar('loss/-objective', L)
        tf.summary.scalar('loss/lr', self.lr)
        self.summaries = tf.summary.merge_all()

    def _build_critic_net(self, s_inputs, scope):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, **initializer_helper)
            v = tf.layers.dense(l, 1, **initializer_helper)

        return v

    def _build_actor_net(self, s_inputs, scope):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, **initializer_helper)

            mu = tf.layers.dense(l, 32, tf.nn.relu, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, **initializer_helper)
            sigma = tf.layers.dense(l, 32, tf.nn.relu, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.softplus, **initializer_helper)

            mu, sigma = mu * self.a_bound, sigma

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        return norm_dist

    def train(self, s, a, discounted_r, mean_reward, iteration):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(discounted_r.shape) == 2
        assert s.shape[0] == a.shape[0] == discounted_r.shape[0]

        self._cache_variables_cachable()
        self._cache_mean_reward_and_judge_converged(mean_reward)

        if iteration % self.save_per_iter == 0:
            self.saver.save(iteration + self.init_iteration)

        if self.summary_writer is not None:
            summaries = self.sess.run(self.summaries, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_discounted_r: discounted_r
            })
            self.summary_writer.add_summary(summaries, iteration + self.init_iteration)

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
