import numpy as np
import tensorflow as tf
import sys
import time

sys.path.append('../..')
from mlagents.envs import UnityEnvironment
from util.memory import Memory
from util.saver import Saver
from ddpg import Actor, Critic


BATCH_SIZE = 32
MEMORY_MAX_SIZE = 3200
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
action_dim = brain_params.vector_action_space_size[0]


variance = 2.
with tf.Session() as sess:
    with tf.device('/cpu:0'):
        memory = Memory(BATCH_SIZE, MEMORY_MAX_SIZE)
        actor = Actor(sess,
                      state_dim,
                      action_dim,
                      lr=0.00005,
                      tau=0.01)
        critic = Critic(sess,
                        actor.pl_s,
                        actor.pl_s_,
                        actor.a,
                        actor.a_,
                        gamma=0.99,
                        lr=0.00005,
                        tau=0.01)

        actor.generate_gradients(critic.get_gradients())

    saver = Saver('model', sess)
    saver.restore_or_init()

    if train_mode:
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        summary_writer = tf.summary.FileWriter(f'log', sess.graph)
        summary_writer.close()
        summary_writer = tf.summary.FileWriter(f'log/{time_str}')

    done = False
    for episode in range(ITER_MAX):
        rewards_sum = 0
        steps_n = 0

        if not done:
            brain_info = env.reset(train_mode=train_mode)[default_brain_name]
            state = brain_info.vector_observations[0]
        else:
            done = False

        while not done and steps_n < MAX_STEPS:
            if train_mode:
                action = actor.choose_action(state, variance)
            else:
                action = actor.choose_action(state)

            brain_info = env.step({
                default_brain_name: action[np.newaxis, :],
            })[default_brain_name]
            state_ = brain_info.vector_observations[0]
            reward = brain_info.rewards[0]

            rewards_sum += reward
            done = brain_info.local_done[0]
            steps_n += 1

            if train_mode:
                memory.store_transition(state, action, [reward], state_, [done])
                if memory.isFull:
                    b_s, b_a, b_r, b_s_, b_done = memory.get_mini_batches()
                    critic.learn(b_s, b_a, b_r, b_s_, b_done)
                    actor.learn(b_s)

            state = state_

        if train_mode:
            summaries = tf.Summary(value=[
                tf.Summary.Value(tag='reward', simple_value=rewards_sum)
            ])
            summary_writer.add_summary(summaries, episode)
            print(f'episode {episode}, rewards {rewards_sum:.2f}, steps {steps_n}, hitted {reward > 0}, variance {variance:.3f}')
        else:
            print(f'episode {episode}, rewards {rewards_sum:.2f}, steps {steps_n}, hitted {reward > 0}')

        if train_mode:
            if memory.isFull and reward > 0:
                variance *= 0.999
            if episode % 500 == 0:
                saver.save(episode)
