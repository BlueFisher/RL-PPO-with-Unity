import sys
import tensorflow as tf

sys.path.append('../../ppo')
from ppo_base import PPO_Base, initializer_helper

initializer_helper = {
    'kernel_initializer': tf.truncated_normal_initializer(0, .1),
    'bias_initializer': tf.constant_initializer(.1)
}


class PPO(PPO_Base):
    def _build_net(self, s_inputs, scope, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = tf.layers.dense(s_inputs, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)

            prob_l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(prob_l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(prob_l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.sigmoid, trainable=trainable, **initializer_helper)
            mu, sigma = mu * self.a_bound, sigma * self.variance_bound + .1

            policy = tf.distributions.Normal(loc=mu, scale=sigma)

            v_l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            v_l = tf.layers.dense(v_l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(v_l, 1, trainable=trainable, **initializer_helper)

            variables = tf.get_variable_scope().global_variables()

        return policy, v, variables
