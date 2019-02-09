import sys
import tensorflow as tf

sys.path.append('..')
from ppo.ppo_sep_critic_base import PPO_Base, Critic_Base, initializer_helper


class Critic(Critic_Base):
    def _build_net(self, s_inputs, scope, trainable, reuse=False):
        with tf.variable_scope(scope):
            l = tf.layers.dense(self.pl_s, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(l, 1, trainable=trainable, **initializer_helper)

        return v


class PPO(PPO_Base):
    def _build_net(self, s_inputs, scope, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 128, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)

            mu = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(l, 32, tf.nn.relu, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.sigmoid, trainable=trainable, **initializer_helper)

            mu, sigma = mu * self.a_bound, sigma * self.variance_bound + .1

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

            variables = tf.get_variable_scope().global_variables()

        return norm_dist, variables
