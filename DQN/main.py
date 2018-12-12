import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from dqn import DQN


GAMMA = 0.99
BATCH_SIZE = 512
ITER_MAX = 10000
MAX_STEPS = 500


train_mode = 'run' not in sys.argv
if train_mode:
    print('Training Mode')
else:
    print('Inference Mode')

env = UnityEnvironment()

default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = len(brain_params.vector_action_space_size)
actions = []
for des in brain_params.vector_action_descriptions:
    action = des.split(',')
    action = [float(i) for i in action]
    actions.append(np.array(action))


with tf.Session() as sess:
    dqn = DQN(
        sess=sess,
        s_dim=state_dim,
        a_dim=action_dim,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=0.0001,
        epsilon=0.1,
        replace_target_iter=100)
    saver = tf.train.Saver()
    if os.path.exists('tmp/checkpoint'):
        saver.restore(sess, "tmp/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())

    for iteration in range(ITER_MAX):
        if train_mode:
            rewards_sum = 0
            hitted_sum = 0
            done = False
            steps_n = 0
            brain_info = env.reset(train_mode=train_mode)[default_brain_name]

            state = brain_info.vector_observations[0]
            while not done and steps_n < MAX_STEPS:
                action_i = dqn.choose_action(state)
                action = actions[action_i]
                assert len(action.shape) == 1
                brain_info = env.step({
                    default_brain_name: action[np.newaxis, :]
                })
                state_ = brain_info.vector_observations[0]
                reward = brain_info.rewards[0]
                done = brain_info.local_done[0]

                dqn.store_transition_and_learn(state, action_i, np.array([reward]), state_, np.array(float(done)))
                steps_n += 1
                rewards_sum += reward

            print(f'episode {iteration},rewards {rewards_sum:.2f},steps {steps_n},succeeded {done}')
