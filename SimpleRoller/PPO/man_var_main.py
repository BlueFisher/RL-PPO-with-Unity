import numpy as np
import tensorflow as tf
import sys
import time

sys.path.append('../..')
from mlagents.envs import UnityEnvironment
from util.saver import Saver
from man_var_ppo import PPO


GAMMA = 0.99
BATCH_SIZE = 512
ITER_MAX = 10000
MAX_STEPS = 500
LEARNING_RATE = 0.00005
INFERENCE_MODE_VARIANCE = 0.2


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


def simulate():
    hitted_sum = 0
    steps_n = 0
    brain_info = env.reset(train_mode=train_mode)[default_brain_name]

    dones = [False] * len(brain_info.agents)
    last_states_ = [0] * len(brain_info.agents)
    trans_all = [[] for _ in range(len(brain_info.agents))]
    rewards_sum = [0] * len(brain_info.agents)
    states = brain_info.vector_observations
    while False in dones and steps_n < MAX_STEPS:
        if train_mode:
            actions = ppo.choose_action(states)
        else:
            actions = ppo.choose_action(states, INFERENCE_MODE_VARIANCE)

        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        states_ = brain_info.vector_observations

        for i in range(len(brain_info.agents)):
            if train_mode:
                last_states_[i] = states_[i]
                trans_all[i].append([states[i], actions[i], np.array([rewards[i]]), local_dones[i]])

            if not dones[i]:
                rewards_sum[i] += rewards[i]
            if rewards[i] > 0:
                hitted_sum += 1

            dones[i] = dones[i] or local_dones[i]
        steps_n += 1
        states = states_

    if train_mode:
        for i in range(len(brain_info.agents)):
            trans = trans_all[i]
            v_state_ = ppo.get_v(last_states_[i])
            for tran in trans[::-1]:  # state, action, reward, done
                if tran[3]:
                    v_state_ = 0
                v_state_ = tran[2] + GAMMA * v_state_
                tran[2] = v_state_
        trans_with_discounted_rewards_all = []
        for trans in trans_all:
            trans_with_discounted_rewards_all += trans

        return trans_with_discounted_rewards_all, rewards_sum, hitted_sum
    else:
        return None, rewards_sum, hitted_sum


with tf.Session() as sess:
    ppo = PPO(sess,
              state_dim,
              action_dim,
              action_bound,
              epsilon=0.2,
              lr=LEARNING_RATE,
              K=10)

    saver = Saver('model/model_man_var', sess)
    last_iteration = saver.restore_or_init()

    if train_mode:
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        summary_writer = tf.summary.FileWriter(f'log/man_var', sess.graph)
        summary_writer.close()
        summary_writer = tf.summary.FileWriter(f'log/man_var/{time_str}')

    for iteration in range(last_iteration, last_iteration + ITER_MAX + 1):
        trans_with_discounted_r, rewards_sum, hitted = simulate()
        mean_reward = sum(rewards_sum) / len(rewards_sum)

        if train_mode:
            if hitted > 1:
                ppo.decrease_sigma()

            for i in range(0, len(trans_with_discounted_r), BATCH_SIZE):
                batch = trans_with_discounted_r[i:i + BATCH_SIZE]
                s, a, discounted_r, *_ = [np.array(e) for e in zip(*batch)]
                ppo.train(s, a, discounted_r)

            s, a, discounted_r, *_ = [np.array(e) for e in zip(*trans_with_discounted_r)]
            summaries = ppo.get_summaries(s, a, discounted_r)
            summary_writer.add_summary(summaries, iteration)
            summaries = tf.Summary(value=[
                tf.Summary.Value(tag='reward/mean', simple_value=mean_reward),
                tf.Summary.Value(tag='reward/max', simple_value=max(rewards_sum)),
                tf.Summary.Value(tag='reward/min', simple_value=min(rewards_sum))
            ])
            summary_writer.add_summary(summaries, iteration)

            if iteration % 500 == 0:
                saver.save(iteration)

            if iteration % 20 == 0:
                ppo.print_test(s)

        print(f'iter {iteration}, rewards {mean_reward:.2f}, hitted {hitted}')
