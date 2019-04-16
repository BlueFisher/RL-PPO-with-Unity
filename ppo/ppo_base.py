import time
import collections
import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')
from util.saver import Saver

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0, .1),
    'bias_initializer': tf.constant_initializer(.1)
}


class PPO_Base(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 saver_model_path='model',
                 save_per_iter=1000,
                 summary_path='log',
                 summary_name=None,
                 write_summary_graph=False,
                 seed=None,
                 init_td_threshold=0.0,
                 td_threshold_decay_steps=100,
                 td_threshold_rate=0.5,
                 batch_size=2048,
                 variance_bound=1.,
                 addition_objective=False,
                 beta=0.001,  # entropy coefficient
                 epsilon=0.2,  # clip bound
                 combine_ratio=1,
                 init_lr=5e-5,
                 min_lr=1e-5,
                 decay_steps=50,
                 decay_rate=0.9,
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
        self.save_per_iter = save_per_iter

        with self.graph.as_default():
            if seed is not None:
                tf.random.set_random_seed(seed)
            self._build_model(addition_objective, combine_ratio, beta, epsilon,
                              init_td_threshold, td_threshold_decay_steps, td_threshold_rate,
                              init_lr, min_lr, decay_steps, decay_rate)
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

    def _build_model(self, addition_objective, combine_ratio, beta, epsilon,
                     init_td_threshold, td_threshold_decay_steps, td_threshold_rate,
                     init_lr, min_lr, decay_steps, decay_rate):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state')
        self.policy, self.v, policy_v_variables = self._build_net(self.pl_s, 'actor_critic', True)
        old_policy, old_v, old_policy_v_variables = self._build_net(self.pl_s, 'old_actor_critic', False)

        with tf.name_scope('objective_and_value_function_loss'):
            self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='action')
            self.pl_advantage = tf.placeholder(tf.float32, shape=(None, 1), name='advantage')
            self.pl_discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_reward')

            self.policy_prob = self.policy.prob(self.pl_a)
            if addition_objective:
                ratio = self.policy_prob - old_policy.prob(self.pl_a)
                L_clip = tf.math.reduce_mean(tf.math.minimum(
                    ratio * self.pl_advantage,  # surrogate objective
                    tf.clip_by_value(ratio, -epsilon, epsilon) * self.pl_advantage
                ), name='clipped_objective')
            else:
                ratio = self.policy_prob / old_policy.prob(self.pl_a)
                L_clip = tf.math.reduce_mean(tf.math.minimum(
                    ratio * self.pl_advantage,  # surrogate objective
                    tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * self.pl_advantage
                ), name='clipped_objective')

            L_vf = tf.reduce_mean(tf.square(self.pl_discounted_r - self.v), name='value_function_loss')
            S = tf.reduce_mean(self.policy.entropy(), name='entropy')

        self.choose_action_op = tf.squeeze(self.policy.sample(1), axis=0)

        with tf.name_scope('optimizer'):
            self.global_iter = tf.get_variable('global_iter', shape=(), initializer=tf.constant_initializer(0), trainable=False)
            self.lr = tf.math.maximum(tf.train.exponential_decay(learning_rate=init_lr,
                                                                 global_step=self.global_iter,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True), min_lr)

            self.td_threshold = tf.train.exponential_decay(init_td_threshold,
                                                           global_step=self.global_iter,
                                                           decay_steps=td_threshold_decay_steps,
                                                           decay_rate=td_threshold_rate,
                                                           staircase=True)
            if combine_ratio > 0:
                L = L_clip - combine_ratio * L_vf + beta * S
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-L)
            else:
                L = L_clip + beta * S
                self.train_op = [tf.train.AdamOptimizer(self.lr).minimize(-L),
                                 tf.train.AdamOptimizer(self.lr).minimize(L_vf)]

        self.update_variables_op = [tf.assign(r, v) for r, v in
                                    zip(old_policy_v_variables, policy_v_variables)]

        tf.summary.scalar('loss/value_function', L_vf)
        tf.summary.scalar('loss/-entropy', S)
        tf.summary.scalar('loss/lr', self.lr)
        self.summaries = tf.summary.merge_all()

    def _build_net(self, s_inputs, trainable):
        # return policy, v, variables
        raise Exception('PPO_Base._build_net not implemented')

    def get_td_error(self, s, r, s_, done):
        assert len(s.shape) == 2
        assert len(r.shape) == 2
        assert len(s_.shape) == 2
        assert len(done.shape) == 2

        return self.sess.run(self.td_error, {
            self.pl_s: s,
            self.pl_r: r,
            self.pl_s_: s_,
            self.pl_done: done
        })

    def get_v(self, s):
        assert len(s.shape) == 2

        return self.sess.run(self.v, {
            self.pl_s: s
        })

    def choose_action(self, s):
        assert len(s.shape) == 2

        a = self.sess.run(self.choose_action_op, {
            self.pl_s: s
        })

        # row, col = a.shape
        # for i in range(col):
        #     action = a[:, i]
        #     bound = self.a_bound[i]
        #     width = bound * 2
        #     for j in range(row):
        #         if bound < action[j] < bound + width:
        #             a[j, i] = width - action[j]
        #         elif -bound - width < action[j] < -bound:
        #             a[j, i] = -width - action[j]
        #         elif action[j] <= -bound - width or action[j] >= bound + width:
        #             a[j, i] = np.random.rand(1) * width - bound

        if np.isnan(np.min(a)):
            print('WARNING! NAN IN ACTIONS')

        return np.clip(a, -self.a_bound, self.a_bound)

    def get_policy(self, s):
        assert len(s.shape) == 2

        return self.sess.run([self.policy.loc, self.policy.scale], {
            self.pl_s: s
        })

    def write_constant_summaries(self, constant_summaries, iteration):
        if self.summary_writer is not None:
            summaries = tf.Summary(value=[tf.Summary.Value(tag=i['tag'],
                                                           simple_value=i['simple_value'])
                                          for i in constant_summaries])
            self.summary_writer.add_summary(summaries, iteration + self.init_iteration)

    def get_not_zero_prob_bool_mask(self, s, a):
        policy_prob = self.sess.run(self.policy_prob, {
            self.pl_s: s,
            self.pl_a: a
        })
        bool_mask = ~np.any(policy_prob <= 1.e-7, axis=1)
        return bool_mask

    def train(self, s, a, adv, discounted_r, iteration):
        assert len(s.shape) == 2
        assert len(a.shape) == 2
        assert len(adv.shape) == 2
        assert len(discounted_r.shape) == 2
        assert s.shape[0] == a.shape[0] == adv.shape[0] == discounted_r.shape[0]

        global_iter = iteration + self.init_iteration
        self.global_iter.load(global_iter, self.sess)

        td_error = np.square(self.get_v(s) - discounted_r)
        bool_mask = np.all(td_error > self.sess.run(self.td_threshold), axis=1)
        if not np.all(bool_mask == False):
            s, a, adv, discounted_r = s[bool_mask], a[bool_mask], adv[bool_mask], discounted_r[bool_mask]

        self.sess.run(self.update_variables_op)  # TODO

        if iteration % self.save_per_iter == 0:
            self.saver.save(global_iter)

        if self.summary_writer is not None:
            summaries = self.sess.run(self.summaries, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_advantage: adv,
                self.pl_discounted_r: discounted_r
            })
            self.summary_writer.add_summary(summaries, global_iter)

        for i in range(0, s.shape[0], self.batch_size):
            _s, _a, _adv, _discounted_r = (s[i:i + self.batch_size],
                                           a[i:i + self.batch_size],
                                           adv[i:i + self.batch_size],
                                           discounted_r[i:i + self.batch_size])
            for _ in range(self.epoch_size):
                self.sess.run(self.train_op, {
                    self.pl_s: _s,
                    self.pl_a: _a,
                    self.pl_advantage: _adv,
                    self.pl_discounted_r: _discounted_r
                })

    def dispose(self):
        self.summary_writer.close()
        self.sess.close()
