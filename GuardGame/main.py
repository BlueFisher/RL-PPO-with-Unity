import numpy as np
import tensorflow as tf
import sys

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from util.memory import Memory
from util.saver import Saver
from ddpg import Actor, Critic


BATCH_SIZE = 64
MEMORY_MAX_SIZE = 3200
ITER_MAX = 10000
MAX_STEPS = 500

train_mode = 'run' not in sys.argv
if train_mode:
    print('Training Mode')
else:
    print('Inference Mode')

env = UnityEnvironment()

attack_brain_name = 'AttackBrain'
guard_brain_name = 'GuardBrain'
assert len(env.brain_names) == 2
assert attack_brain_name in env.brain_names
assert guard_brain_name in env.brain_names

brain_params = env.brains
state_dims = {
    attack_brain_name: brain_params[attack_brain_name].vector_observation_space_size,
    guard_brain_name: brain_params[guard_brain_name].vector_observation_space_size
}
action_dims = {
    attack_brain_name: brain_params[attack_brain_name].vector_action_space_size[0],
    guard_brain_name: brain_params[guard_brain_name].vector_action_space_size[0]
}
action_bounds = {
    attack_brain_name: np.array([float(i) for i in brain_params[attack_brain_name].vector_action_descriptions]),
    guard_brain_name: np.array([float(i) for i in brain_params[guard_brain_name].vector_action_descriptions])
}


class Agent(object):
    def __init__(self, sess, scope, batch_size, memory_max_size, state_dim, action_dim, action_bound):
        with tf.variable_scope(scope):
            self.memory = Memory(batch_size, memory_max_size)
            self.actor = Actor(sess,
                               state_dim,
                               action_dim,
                               action_bound,
                               lr=0.0001,
                               tau=0.01)
            self.critic = Critic(sess,
                                 self.actor.pl_s,
                                 self.actor.pl_s_,
                                 self.actor.a,
                                 self.actor.a_,
                                 gamma=0.99,
                                 lr=0.0001,
                                 tau=0.01)
            self.actor.generate_gradients(self.critic.get_gradients())
            self.variance = 3.

    def choose_action(self, state, use_variance=False):
        if use_variance:
            return self.actor.choose_action(state, self.variance)
        else:
            return self.actor.choose_action(state)

    def learn(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, [reward], state_, [done])
        if self.memory.isFull:
            b_s, b_a, b_r, b_s_, b_done = self.memory.get_mini_batches()
            self.critic.learn(b_s, b_a, b_r, b_s_, b_done)
            self.actor.learn(b_s)

    def decrease_variance(self):
        if self.memory.isFull:
            self.variance *= 0.9999


with tf.Session() as sess:
    attack_agent = Agent(sess, 'attack', BATCH_SIZE, MEMORY_MAX_SIZE,
                         state_dims[attack_brain_name],
                         action_dims[attack_brain_name],
                         action_bounds[attack_brain_name])
    guard_agent = Agent(sess, 'guard', BATCH_SIZE, MEMORY_MAX_SIZE,
                        state_dims[guard_brain_name],
                        action_dims[guard_brain_name],
                        action_bounds[guard_brain_name])

    saver = Saver('model', sess)
    saver.restore_or_init()

    for episode in range(ITER_MAX):
        dones_control = [False] * 2
        done = False
        rewards_sum = {
            attack_brain_name: 0,
            guard_brain_name: 0
        }
        steps_n = 0

        brain_infos = env.reset(train_mode=train_mode)
        states = {
            attack_brain_name: brain_infos[attack_brain_name].vector_observations[0],
            guard_brain_name: brain_infos[guard_brain_name].vector_observations[0]
        }

        while False in dones_control and steps_n < MAX_STEPS:
            actions = {}
            if train_mode:
                actions[attack_brain_name] = attack_agent.choose_action(states[attack_brain_name], True)
                actions[guard_brain_name] = guard_agent.choose_action(states[guard_brain_name], True)
            else:
                actions[attack_brain_name] = attack_agent.choose_action(states[attack_brain_name])
                actions[guard_brain_name] = guard_agent.choose_action(states[guard_brain_name])

            brain_infos = env.step({
                attack_brain_name: actions[attack_brain_name][np.newaxis, :],
                guard_brain_name: actions[guard_brain_name][np.newaxis, :]
            })

            states_ = {
                attack_brain_name: brain_infos[attack_brain_name].vector_observations[0],
                guard_brain_name: brain_infos[guard_brain_name].vector_observations[0]
            }
            rewards = {
                attack_brain_name: brain_infos[attack_brain_name].rewards[0],
                guard_brain_name: brain_infos[guard_brain_name].rewards[0]
            }
            dones = {
                attack_brain_name: brain_infos[attack_brain_name].local_done[0],
                guard_brain_name: brain_infos[guard_brain_name].local_done[0]
            }

            rewards_sum[attack_brain_name] += rewards[attack_brain_name]
            rewards_sum[guard_brain_name] += rewards[guard_brain_name]
            dones_control[0] = dones_control[0] or dones[attack_brain_name]
            dones_control[1] = dones_control[1] or dones[guard_brain_name]
            steps_n += 1

            if train_mode:
                attack_agent.learn(states[attack_brain_name],
                                   actions[attack_brain_name],
                                   [rewards[attack_brain_name]],
                                   states_[attack_brain_name],
                                   [dones[attack_brain_name]])
                guard_agent.learn(states[guard_brain_name],
                                  actions[guard_brain_name],
                                  [rewards[guard_brain_name]],
                                  states_[guard_brain_name],
                                  [dones[guard_brain_name]])

            states = states_

        print(f'episode {episode}, steps: {steps_n}')
        print(f'rewards a:{rewards_sum[attack_brain_name]:.2f} g:{rewards_sum[guard_brain_name]:.2f}, steps {steps_n}')
        if train_mode:
            print(f'variance a:{attack_agent.variance:.3f} g:{guard_agent.variance}')

        if train_mode:
            attack_agent.decrease_variance()
            guard_agent.decrease_variance()
            if episode % 500 == 0:
                saver.save(episode)
