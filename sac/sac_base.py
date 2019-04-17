import time
import collections
import sys

import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

sys.path.append('..')
from util.saver import Saver

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0, .1),
    'bias_initializer': tf.constant_initializer(.1)
}


class SAC_Base(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 saver_model_path='model',
                 summary_path='log',
                 summary_name=None,
                 write_summary_graph=False,
                 seed=None,
                 gamma=0.99,
                 tau=0.005,
                 write_summary_per_step=20,
                 update_target_per_step=1,
                 init_lr=3e-4,
                 min_lr=0,
                 decay_steps=50,
                 decay_rate=1.,
                 use_priority=False,
                 batch_size=256,
                 max_replay_buffer_size=25600):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                               graph=self.graph)

        self.use_priority = use_priority
        if self.use_priority:
            self.replay_buffer = PrioritizedReplayBuffer(batch_size, max_replay_buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(batch_size, max_replay_buffer_size)

        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_bound = action_bound

        self.write_summary_per_step = write_summary_per_step
        self.update_target_per_step = update_target_per_step

        with self.graph.as_default():
            if seed is not None:
                tf.random.set_random_seed(seed)

            self._build_model(gamma, tau,
                              init_lr, min_lr, decay_steps, decay_rate)

            self.saver = Saver(saver_model_path, self.sess)
            self.init_iteration = self.saver.restore_or_init()

            self.sess.run(self.update_target_hard_op)

            self.summary_writer = None
            if summary_path is not None:
                if summary_name is None:
                    summary_name = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

                if write_summary_graph:
                    writer = tf.summary.FileWriter(f'{summary_path}/{summary_name}', self.graph)
                    writer.close()
                self.summary_writer = tf.summary.FileWriter(f'{summary_path}/{summary_name}')

    def _build_model(self, gamma, tau,
                     init_lr, min_lr, decay_steps, decay_rate):
        self.pl_s = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state')
        self.pl_a = tf.placeholder(tf.float32, shape=(None, self.a_dim), name='action')
        self.pl_r = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
        self.pl_s_ = tf.placeholder(tf.float32, shape=(None, self.s_dim), name='state_')
        self.pl_done = tf.placeholder(tf.float32, shape=(None, 1), name='done')

        self.pl_is = tf.placeholder(tf.float32, shape=(None, 1), name='importance_ratio')

        log_alpha = tf.get_variable('alpha', shape=(), initializer=tf.constant_initializer(-4.6))
        alpha = tf.exp(log_alpha)

        policy, self.action_sampled, policy_variables = self._build_policy_net(self.pl_s, 'policy')
        policy_next, action_next_sampled, policy_next_variables = self._build_policy_net(self.pl_s_, 'policy', reuse=True)

        q1, q1_variables = self._build_q_net(self.pl_s, self.pl_a, 'q1')
        q1_for_gradient, _ = self._build_q_net(self.pl_s, self.action_sampled, 'q1', reuse=True)
        q1_target, q1_target_variables = self._build_q_net(self.pl_s_, action_next_sampled, 'q1_target', trainable=False)

        q2, q2_variables = self._build_q_net(self.pl_s, self.pl_a, 'q2')
        q2_target, q2_target_variables = self._build_q_net(self.pl_s_, action_next_sampled, 'q2_target', trainable=False)

        y = self.pl_r + gamma * (1 - self.pl_done) * (tf.minimum(q1_target, q2_target) - alpha * policy_next.log_prob(action_next_sampled))
        y = tf.stop_gradient(y)

        if self.use_priority:
            L_q1 = tf.reduce_mean(tf.squared_difference(q1, y) * self.pl_is)
            L_q2 = tf.reduce_mean(tf.squared_difference(q2, y) * self.pl_is)
        else:
            L_q1 = tf.reduce_mean(tf.squared_difference(q1, y))
            L_q2 = tf.reduce_mean(tf.squared_difference(q2, y))
        

        q1_td_error = tf.abs(q1 - y)
        q2_td_error = tf.abs(q2 - y)
        self.td_error = tf.reduce_mean(tf.concat([q1_td_error, q2_td_error], axis=1),
                                       axis=1, keepdims=True)

        L_policy = tf.reduce_mean(alpha * policy.log_prob(self.action_sampled) - q1_for_gradient)

        entropy = policy.entropy()
        L_alpha = -log_alpha * policy.log_prob(self.action_sampled) + log_alpha * 2

        with tf.name_scope('optimizer'):
            self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0), trainable=False)
            lr = tf.math.maximum(tf.train.exponential_decay(learning_rate=init_lr,
                                                            global_step=self.global_step,
                                                            decay_steps=decay_steps,
                                                            decay_rate=decay_rate,
                                                            staircase=True), min_lr)

            self.train_q_ops = [tf.train.AdamOptimizer(lr).minimize(L_q1,
                                                                    var_list=q1_variables),
                                tf.train.AdamOptimizer(lr).minimize(L_q2,
                                                                    var_list=q2_variables)]
            self.train_policy_op = tf.train.AdamOptimizer(lr).minimize(L_policy,
                                                                       global_step=self.global_step,
                                                                       var_list=policy_variables)
            self.train_alpha_op = tf.train.AdamOptimizer(lr).minimize(L_alpha,
                                                                      var_list=[log_alpha])

        self.update_target_hard_op = [tf.assign(t, e) for t, e in
                                      zip(q1_target_variables + q2_target_variables, q1_variables + q2_variables)]
        self.update_target_op = [tf.assign(t, tau * e + (1 - tau) * t) for t, e in
                                 zip(q1_target_variables + q2_target_variables, q1_variables + q2_variables)]

        tf.summary.scalar('loss/Q1', L_q1)
        tf.summary.scalar('loss/Q2', L_q2)
        # tf.summary.scalar('loss/lr', lr)
        tf.summary.scalar('loss/policy', L_policy)
        tf.summary.scalar('loss/entropy', tf.reduce_mean(entropy))
        tf.summary.scalar('loss/alpha', alpha)
        self.summaries = tf.summary.merge_all()

    def _build_q_net(self, s_input, a_input, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_q_net not implemented')
        # return q, variables

    def _build_policy_net(self, s_inputs, scope, trainable=True, reuse=False):
        raise Exception('SAC_Base._build_policy_net not implemented')
        # return policy, action_sampled, variables

    def choose_action(self, s):
        assert len(s.shape) == 2

        a = self.sess.run(self.action_sampled, {
            self.pl_s: s,
        })

        return a

    def save_model(self, iteration):
        self.saver.save(iteration + self.init_iteration)

    def write_constant_summaries(self, constant_summaries, iteration):
        if self.summary_writer is not None:
            summaries = tf.Summary(value=[tf.Summary.Value(tag=i['tag'],
                                                           simple_value=i['simple_value'])
                                          for i in constant_summaries])
            self.summary_writer.add_summary(summaries, iteration + self.init_iteration)

    def train(self, s, a, r, s_, done):
        assert len(s.shape) == 2

        global_step = self.sess.run(self.global_step)

        self.replay_buffer.add_sample(s, a, r, s_, done)

        if self.use_priority:
            points, (s, a, r, s_, done), importance_ratio = self.replay_buffer.random_batch()
        else:
            s, a, r, s_, done = self.replay_buffer.random_batch()

        if global_step % self.update_target_per_step == 0:
            self.sess.run(self.update_target_op)

        if global_step % self.write_summary_per_step == 0:
            summaries = self.sess.run(self.summaries, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_r: r,
                self.pl_s_: s_,
                self.pl_done: done,
                self.pl_is: np.zeros((1,1)) if not self.use_priority else importance_ratio
            })
            self.summary_writer.add_summary(summaries, global_step)

        if self.replay_buffer.is_lg_batch_size:
            self.sess.run(self.train_q_ops, {
                self.pl_s: s,
                self.pl_a: a,
                self.pl_r: r,
                self.pl_s_: s_,
                self.pl_done: done,
                self.pl_is: np.zeros((1,1)) if not self.use_priority else importance_ratio
            })

            self.sess.run([self.train_policy_op, self.train_alpha_op], {
                self.pl_s: s,
            })

            if self.use_priority:
                td_error = self.sess.run(self.td_error, {
                    self.pl_s: s,
                    self.pl_a: a,
                    self.pl_r: r,
                    self.pl_s_: s_,
                    self.pl_done: done
                })

                self.replay_buffer.update(points, np.squeeze(td_error))


if __name__ == '__main__':
    sac = SAC(3, 2, np.array([1, 1]))

    sac.train(np.array([[1., 1., 2.]]),
              np.array([[2., 2.]]),
              np.array([[1.]]),
              np.array([[1., 1., 3.]]),
              np.array([[0]]),
              10)