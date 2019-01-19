import numpy as np
import tensorflow as tf

from .ppo_base import PPO_Base, initializer_helper


class PPO_SEP_NN(PPO_Base):
    def _build_net(self, s_inputs, scope, trainable):
        with tf.variable_scope(scope):
            policy, policy_variables = self._build_actor_net(s_inputs, 'actor', trainable)
            v, v_variables = self._build_critic_net(s_inputs, 'critic', trainable)

        return policy, v, policy_variables + v_variables

    def _build_critic_net(self, s_inputs, scope, trainable):
        # return v, variables
        raise Exception('PPO_SEP_NN._build_critic_net not implemented')

    def _build_actor_net(self, s_inputs, scope, trainable):
        # return policy, variables
        raise Exception('PPO_SEP_NN._build_critic_net not implemented')
