import numpy as np
import tensorflow as tf
import sys
import itertools

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from util.saver import Saver
from dqn import DQN


GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_MAX_SIZE = 3200
LEARNING_RATE = 0.0001
EPSILON = 0.1
REPLACE_TARGET_ITER = 100
ITER_MAX = 10000
MAX_STEPS = 200


train_mode = 'run' not in sys.argv
if train_mode:
    print('Training Mode')
else:
    print('Inference Mode')

env = UnityEnvironment()

default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
assert brain_params.vector_action_space_type == 'discrete'
actions = list(itertools.product(*[i.split(',')
                                   for i in brain_params.vector_action_descriptions]
                                 ))
actions = [np.array(i, dtype=float) for i in actions]
print(actions)
action_dim = len(actions)


with tf.Session() as sess:
    dqn = DQN(
        sess=sess,
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=BATCH_SIZE,
        memory_max_size=MEMORY_MAX_SIZE,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        epsilon=EPSILON,
        replace_target_iter=REPLACE_TARGET_ITER)

    saver = Saver('model', sess)
    saver.restore_or_init()

    for episode in range(ITER_MAX):
        rewards_sum = 0
        done = False
        steps_n = 0
        brain_info = env.reset(train_mode=train_mode)[default_brain_name]

        state = brain_info.vector_observations[0]
        while not done and steps_n < MAX_STEPS:
            action_i = dqn.choose_action(state, train_mode)
            action = actions[action_i]
            assert len(action.shape) == 1
            brain_info = env.step({
                default_brain_name: action[np.newaxis, :]
            })[default_brain_name]
            state_ = brain_info.vector_observations[0]
            reward = brain_info.rewards[0]
            done = brain_info.local_done[0]

            if train_mode:
                dqn.store_transition_and_learn(state, action_i, [reward],
                                               state_, [float(done)])
            steps_n += 1
            rewards_sum += reward
            state = state_

        print(f'episode {episode}, rewards {rewards_sum: .2f}, steps {steps_n}, hitted {reward > 0}')

        if train_mode and episode % 20 == 0:
            dqn.test([state])
        if train_mode and episode % 1000 == 0:
            saver.save(episode)
