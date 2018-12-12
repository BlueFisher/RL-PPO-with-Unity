import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from std_ppo import PPO


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
action_dim = brain_params.vector_action_space_size[0]
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])


def simulate_training():
    rewards_sum = 0
    hitted_sum = 0
    brain_info = env.reset(train_mode=True)[default_brain_name]

    dones = [False] * len(brain_info.agents)
    last_states_ = [0] * len(brain_info.agents)
    trans_all_agents = [[] for _ in range(len(brain_info.agents))]
    states = brain_info.vector_observations
    while False in dones:
        actions = ppo.choose_action(states)
        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        states_ = brain_info.vector_observations

        for i in range(len(brain_info.agents)):
            last_states_[i] = states_[i]
            trans_all_agents[i].append([states[i], actions[i], np.array([rewards[i]]), local_dones[i]])
            rewards_sum += rewards[i]
            if rewards[i] > 0:
                hitted_sum += 1

            dones[i] = dones[i] or local_dones[i]
        states = states_

    for i in range(len(brain_info.agents)):
        trans = trans_all_agents[i]
        v_state_ = ppo.get_v(last_states_[i])
        for tran in trans[::-1]:  # state, action, reward, done
            if tran[3]:
                v_state_ = 0
            v_state_ = tran[2] + GAMMA * v_state_
            tran[2] = v_state_
    trans_with_discounted_rewards_all = []
    for trans in trans_all_agents:
        trans_with_discounted_rewards_all += trans

    return trans_with_discounted_rewards_all, rewards_sum, hitted_sum


def simulate_inference():
    rewards_sum = 0
    hitted_sum = 0
    brain_info = env.reset(train_mode=False)[default_brain_name]

    dones = [False] * len(brain_info.agents)
    states = brain_info.vector_observations
    while False in dones:
        actions = ppo.choose_action(states)
        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        states_ = brain_info.vector_observations

        for i in range(len(brain_info.agents)):
            rewards_sum += rewards[i]
            if rewards[i] > 0:
                hitted_sum += 1

            dones[i] = dones[i] or local_dones[i]
        states = states_

    return rewards_sum, hitted_sum


with tf.Session() as sess:
    ppo = PPO(sess, state_dim, action_dim, action_bound,
              c1=1, c2=0.001, epsilon=0.2, lr=0.00005, K=10)

    saver = tf.train.Saver()
    if os.path.exists('tmp_std/checkpoint'):
        saver.restore(sess, "tmp_std/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())

    for iteration in range(ITER_MAX):
        if train_mode:
            trans_with_discounted_r, rewards_sum, hitted = simulate_training()
            print(f'iter {iteration}, rewards {rewards_sum:.2f}, hitted {hitted}')

            for i in range(0, len(trans_with_discounted_r), BATCH_SIZE):
                batch = trans_with_discounted_r[i:i + BATCH_SIZE]
                s, a, discounted_r, *_ = [np.array(e) for e in zip(*batch)]
                ppo.train(s, a, discounted_r)

            if iteration % 100 == 0:
                saver.save(sess, 'tmp_std/model.ckpt')

            if iteration % 20 == 0:
                ppo.test(np.array([s[0], s[-1]]))
        else:
            rewards_sum, hitted = simulate_inference()
            print(f'iter {iteration}, rewards {rewards_sum:.2f}, hitted {hitted}')
