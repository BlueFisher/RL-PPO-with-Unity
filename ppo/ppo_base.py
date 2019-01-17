import time
import collections
import os

import numpy as np
import tensorflow as tf

from util.saver import Saver
from util.utils import smooth

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class PPO_Base(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 saver_model_path='model',
                 summary_path='log',
                 summary_name=None,
                 write_summary_graph=False,
                 batch_size=2048,
                 variance_bound=1.,
                 c1=1,
                 c2=0.001,  # entropy coefficient
                 epsilon=0.2,  # clip epsilon
                 init_lr=0.00005,
                 epoch_size=10,
                 save_per_iter=200):  # train K epochs

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
        self.save_per_iter = save_per_iter

        with self.graph.as_default():
            self._build_model(c1, c2, epsilon, init_lr)
            self.saver = Saver(saver_model_path, self.sess)
            self.init_iteration = self.saver.restore_or_init()

            self.summary_writer = None
            if summary_path is not None:
                if summary_name is None:
                    summary_name = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

                if write_summary_graph:
                    writer = tf.summary.FileWriter(f'{summary_path}/{summary_name}', self.graph)
                    writer.close()
                self.summary_writer = tf.summary.FileWriter(f'{summary_path}/{summary_name}')

        self.variables_cached_deque = collections.deque(maxlen=10)
        self.mean_rewards_deque = collections.deque(maxlen=50)

    def _build_model(self, c1, c2, epsilon, init_lr):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state')

        policy, self.v, policy_v_variables = self._build_net(self.pl_s, 'actor_critic', True)
        old_policy, old_v, old_policy_v_variables = self._build_net(self.pl_s, 'old_actor_critic', False)

        self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_reward')

        with tf.name_scope('Objective_and_Value_Function_Loss'):
            advantage = self.pl_discounted_r - old_v

            self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='action')
            ratio = policy.prob(self.pl_a) / old_policy.prob(self.pl_a)
            self.policy_prob = policy.prob(self.pl_a)

            L_clip = tf.math.reduce_mean(tf.math.minimum(
                ratio * advantage,  # surrogate objective
                tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * advantage
            ), name='clipped_objective')
            L_vf = tf.reduce_mean(tf.square(self.pl_discounted_r - self.v), name='value_function_loss')
            S = tf.reduce_mean(policy.entropy(), name='entropy')

            L = L_clip - c1 * L_vf + c2 * S

        self.choose_action_op = tf.squeeze(policy.sample(1), axis=0)

        with tf.name_scope('optimizer'):
            self.lr = tf.get_variable('lr', shape=(), initializer=tf.constant_initializer(init_lr))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-L)

        self.update_variables_op = [tf.assign(r, v) for r, v in
                                    zip(old_policy_v_variables, policy_v_variables)]

        self.variables_cachable = [v for v in tf.global_variables() if v != self.lr]

        tf.summary.scalar('loss/-clipped_objective', L_clip)
        tf.summary.scalar('loss/value_function', L_vf)
        tf.summary.scalar('loss/-entropy', S)
        tf.summary.scalar('loss/-mixed_objective', L)
        tf.summary.scalar('loss/lr', self.lr)
        self.summaries = tf.summary.merge_all()

    def _build_net(self, s_inputs, scope, trainable):
        # return policy, v, variables
        raise Exception('ppo_base._build_net not implemented')

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

    def _restore_variables_cachable(self):
        variables = self.variables_cached_deque[0]
        self.sess.run([tf.assign(r, v) for r, v in
                       zip(self.variables_cachable, variables)])

        self.variables_cached_deque.clear()
        self.variables_cached_deque.append(variables)

    def _cache_variables_cachable(self):
        self.variables_cached_deque.append(self.sess.run(self.variables_cachable))

    def _cache_mean_reward_and_judge_converged(self, mean_reward):
        self.mean_rewards_deque.append(mean_reward)
        if len(self.mean_rewards_deque) == self.mean_rewards_deque.maxlen and \
                np.var(smooth(self.mean_rewards_deque, 0.6)) <= 0.05 and \
                np.mean(self.mean_rewards_deque) >= -0.01:
            print('CONVERGED')
            self._decrease_lr()
            self.mean_rewards_deque.clear()

    def write_constant_summaries(self, constant_summaries, iteration):
        iteration += self.init_iteration
        if self.summary_writer is not None:
            summaries = tf.Summary(value=[tf.Summary.Value(tag=i['tag'],
                                                           simple_value=i['simple_value'])
                                          for i in constant_summaries])
            self.summary_writer.add_summary(summaries, iteration)

    def get_not_zero_prob_bool_mask(self, s, a):
        policy_prob = self.sess.run(self.policy_prob, {
            self.pl_s: s,
            self.pl_a: a
        })
        bool_mask = ~np.any(policy_prob <= 1.e-5, axis=1)
        return bool_mask

    # def get_min_policy_prob(self, s, a):
    #     return np.min(self.sess.run(self.policy_prob, {
    #         self.pl_s: s,
    #         self.pl_a: a,
    #     }))

    def train(self, s, a, discounted_r, mean_reward, iteration):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(discounted_r.shape) == 2
        assert s.shape[0] == a.shape[0] == discounted_r.shape[0]

        self.sess.run(self.update_variables_op)  # TODO

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

        # old_policy_probs = self.sess.run(self.old_policy_prob, {
        #     self.pl_s: s,
        #     self.pl_a: a
        # })
        # idx = ~np.any(old_policy_probs <= 1.e-5, axis=1)
        # print(len(s), np.sum(~idx))
        # s = s[idx]
        # a = a[idx]
        # discounted_r = discounted_r[idx]

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

    def dispose(self):
        self.sess.close()
