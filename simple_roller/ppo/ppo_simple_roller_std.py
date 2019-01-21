import sys

import numpy as np
import tensorflow as tf

sys.path.append('../..')
from ppo.ppo_base import PPO_Base, initializer_helper


class PPO(PPO_Base):
    def _build_net(self, s_inputs, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, trainable=trainable, **initializer_helper)

            prob_l = tf.layers.dense(l, 64, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(prob_l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(prob_l, 64, tf.nn.relu, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.sigmoid, trainable=trainable, **initializer_helper)
            mu, sigma = mu * self.a_bound, sigma * self.variance_bound

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

            v_l = tf.layers.dense(l, 64, tf.nn.relu, trainable=trainable, **initializer_helper)
            v_l = tf.layers.dense(v_l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(v_l, 1, trainable=trainable, **initializer_helper)

            variables = tf.get_variable_scope().global_variables()

        return norm_dist, v, variables
