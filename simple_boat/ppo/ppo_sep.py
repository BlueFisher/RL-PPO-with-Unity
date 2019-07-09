import sys
import tensorflow as tf

sys.path.append('../../ppo')
from ppo_base import PPO_Base,initializer_helper

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0, .1),
    'bias_initializer': tf.constant_initializer(0)
}

class PPO(PPO_Base):
    def _build_net(self, s_inputs, scope, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            policy, policy_variables = self._build_actor_net(s_inputs, 'actor', trainable)
            v, v_variables = self._build_critic_net(s_inputs, 'critic', trainable)

        return policy, v, policy_variables + v_variables

    def _build_critic_net(self, s_inputs, scope, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            v = tf.layers.dense(l, 1, trainable=trainable, **initializer_helper)

            variables = tf.get_variable_scope().global_variables()

        return v, variables

    def _build_actor_net(self, s_inputs, scope, trainable, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            l = tf.layers.dense(s_inputs, 512, tf.nn.relu, trainable=trainable, **initializer_helper)
            l = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)

            mu = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            mu = tf.layers.dense(mu, self.a_dim, tf.nn.tanh, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(l, 256, tf.nn.relu, trainable=trainable, **initializer_helper)
            sigma = tf.layers.dense(sigma, self.a_dim, tf.nn.sigmoid, trainable=trainable, **initializer_helper)

            mu, sigma = mu, sigma * self.variance_bound + .1

            policy = tf.distributions.Normal(loc=mu, scale=sigma)

            variables = tf.get_variable_scope().global_variables()

        return policy, variables

