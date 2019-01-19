import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')
from ppo.ppo_sep_nn_base import PPO_SEP_NN, initializer_helper


class PPO(PPO_SEP_NN):
    def _build_critic_net(self, s_inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(l, 1, trainable=trainable, **initializer_helper)

            variables = tf.get_variable_scope().global_variables()

        return v, variables

    def _build_actor_net(self, s_inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 512, tf.nn.relu, trainable=trainable, **initializer_helper)

            mu = tf.layers.dense(l, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(l, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.sigmoid, trainable=trainable, **initializer_helper)

            mu, sigma = mu * self.a_bound, sigma * self.variance_bound

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

            variables = tf.get_variable_scope().global_variables()

        return norm_dist, variables

    def _cache_mean_reward_and_judge_converged(self, mean_reward):
        self.mean_rewards_deque.append(mean_reward)
        if len(self.mean_rewards_deque) == self.mean_rewards_deque.maxlen and \
                np.mean(self.mean_rewards_deque) >= 40:
            print('CONVERGED')
            self._decrease_lr()
            self.mean_rewards_deque.clear()
