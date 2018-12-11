import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from ddpg import Actor, Critic
from memory import *

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
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])

var = 1.
MAX_STEPS = 5000

tf.reset_default_graph()
with tf.Session() as sess:
    memory = Memory(64, 6400)
    actor = Actor(sess, state_dim, action_dim, action_bound, lr=0.0001, tau=0.01)
    critic = Critic(sess, state_dim, actor.s, actor.s_, actor.a, actor.a_, gamma=0.9, lr=0.0001, tau=0.01)

    actor.generate_gradients(critic.get_gradients())

    saver = tf.train.Saver()
    # tf.train.write_graph(sess.graph_def, 'tmp',
    #                      'raw_graph_def.pb', as_text=False)

    if os.path.exists('tmp/checkpoint'):
        saver.restore(sess, "tmp/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())

    for episode in range(10000):
        done = False

        if train_mode:
            brain_info = env.reset(train_mode=train_mode)[default_brain_name]
            s = brain_info.vector_observations[0]
            steps_n = 0
            rewards_n = 0
            hitted = 0

            while not done and steps_n < MAX_STEPS:
                a = actor.choose_action(s)
                # a (action_dim, )
                a = np.clip(np.random.normal(a, var), -action_bound, action_bound)  # 异策略探索
                brain_info = env.step({
                    default_brain_name: a[np.newaxis, :],
                })[default_brain_name]
                s_ = brain_info.vector_observations[0]
                r = brain_info.rewards[0]

                rewards_n += r
                done = brain_info.local_done[0]
                steps_n += 1
                if r > 0:
                    hitted += 1

                memory.store_transition(s, a, [r], s_, [done])

                if memory.isFull:
                    b_s, b_a, b_r, b_s_, b_done = memory.get_mini_batches()
                    critic.learn(b_s, b_a, b_r, b_s_, b_done)
                    actor.learn(b_s)

                # if episode % 500 == 0 and memory.isFull:
                #     memory.clear()
                #     print('memory cleared')

                s = s_

            print('episode {}\trewards {:.2f}\tsteps {:<2}\thitted {}\tvar {:.2f}'
                  .format(episode, rewards_n, steps_n, hitted, var))

            if memory.isFull and hitted >= 1:
                var *= 0.999
            if episode % 100 == 0:
                saver.save(sess, 'tmp/model.ckpt')
        else:
            steps_n = 1
            brain_info = env.reset(train_mode=train_mode)[default_brain_name]
            s = brain_info.vector_observations[0]
            while not done and steps_n < MAX_STEPS:
                a = actor.choose_action(s)
                brain_info = env.step({
                    default_brain_name: a[np.newaxis, :],
                })[default_brain_name]
                s = brain_info.vector_observations[0]
                r = brain_info.rewards[0]
                done = brain_info.local_done[0]

                steps_n += 1

            if r < 0:
                print('failed')
            if steps_n == 500:
                print('timeout')
