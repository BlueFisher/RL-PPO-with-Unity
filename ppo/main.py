import sys
import getopt
import os
import time

import numpy as np
import tensorflow as tf

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from ppo_sep_nn import PPO

TRAIN_MODE = True
NAME = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
BUILD_PATH = None
PORT = 7000
GAMMA = 0.99
ITER_MAX = 3000
PPO_NUM = 1
AGENTS_NUM_EACH_PPO = 1

try:
    opts, args = getopt.getopt(sys.argv[1:], 'rn:b:p:', ['name=',
                                                         'build=',
                                                         'port==',
                                                         'ppo_num=',
                                                         'agents_num='])
except getopt.GetoptError:
    raise Exception('ARGS ERROR')

for opt, arg in opts:
    if opt == '-r':
        TRAIN_MODE = False
    elif opt in ('-n', '--name'):
        NAME = arg.replace('{time}', NAME)
    elif opt in ('-b', '--build'):
        BUILD_PATH = arg
    elif opt in ('-p', '--port'):
        PORT = int(arg)
    elif opt == '--ppo_num':
        PPO_NUM = int(arg)
    elif opt == '--agents_num':
        AGENTS_NUM_EACH_PPO = int(arg)


def simulate_multippo(env, brain_info, default_brain_name, action_dim, ppos: list):
    len_agents = len(brain_info.agents)
    dones = [False] * len_agents
    trans_all = [[] for _ in range(len_agents)]  # list of all transition each agent
    rewards_all = [0] * len_agents
    hitted_real_all = [0] * len_agents
    hitted_all = [0] * len_agents

    states = brain_info.vector_observations
    while False in dones and not env.global_done:
        actions = np.zeros((len_agents, action_dim))
        for i, ppo in enumerate(ppos):
            actions[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO] \
                = ppo.choose_action(states[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO])

        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        max_reached = brain_info.max_reached
        states_ = brain_info.vector_observations

        for i in range(len(brain_info.agents)):
            if TRAIN_MODE:
                trans_all[i].append([states[i],
                                     actions[i],
                                     np.array([rewards[i]]),
                                     local_dones[i],
                                     max_reached[i],
                                     states_[i]])

            if not dones[i]:
                rewards_all[i] += rewards[i]
                if rewards[i] >= 1:
                    hitted_real_all[i] += 1
            if rewards[i] >= 1:
                hitted_all[i] += 1

            dones[i] = dones[i] or local_dones[i]
        states = states_

    if TRAIN_MODE:
        trans_discounted_all = trans_all
        for i in range(len_agents):
            trans = trans_discounted_all[i]  # all transitions in each agent
            v_state_ = ppos[int(i / AGENTS_NUM_EACH_PPO)].get_v(trans[-1][5])
            for tran in trans[::-1]:  # state, action, reward, done, max_reached, state_
                if tran[4]:  # max_reached
                    v_state_ = ppos[int(i / AGENTS_NUM_EACH_PPO)].get_v(tran[5])
                elif tran[3]:  # not max_reached but done
                    v_state_ = 0
                v_state_ = tran[2] + GAMMA * v_state_
                tran[2] = v_state_

        return brain_info, trans_discounted_all, rewards_all, hitted_all, hitted_real_all
    else:
        return brain_info, None, rewards_all, hitted_all, hitted_real_all


if BUILD_PATH is None:
    env = UnityEnvironment()
else:
    env = UnityEnvironment(file_name=BUILD_PATH,
                           no_graphics=True,
                           base_port=PORT)

# single brain
default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = brain_params.vector_action_space_size[0]
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])

ppos = []
for i in range(PPO_NUM):
    if PPO_NUM > 1:
        name = f'{NAME}/{i+1}'
    else:
        name = NAME
    print('=' * 10, name, '=' * 10)
    ppos.append(PPO(state_dim=state_dim,
                    action_dim=action_dim,
                    action_bound=action_bound,
                    saver_model_path=f'model/{name}',
                    summary_name=name,
                    write_summary_graph=True))

reset_config = {
    'copy': AGENTS_NUM_EACH_PPO * PPO_NUM
}

brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
for iteration in range(ITER_MAX + 1):
    if env.global_done:
        brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
    brain_info, trans_discounted_all, rewards_all, hitted_all, hitted_real_all = \
        simulate_multippo(env, brain_info, default_brain_name, action_dim, ppos)

    for i, ppo in enumerate(ppos):
        rewards = rewards_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO]
        mean_reward = sum(rewards) / AGENTS_NUM_EACH_PPO
        hitted = sum(hitted_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO])
        hitted_real = sum(hitted_real_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO])

        print(f'ppo {i+1}, iter {iteration}, rewards {mean_reward:.2f}, hitted {hitted}, hitted_real {hitted_real}')

        if TRAIN_MODE:
            trans_discounted = []
            for trans_discounted_j in trans_discounted_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO]:
                trans_discounted += trans_discounted_j
            ppo.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': mean_reward},
                {'tag': 'reward/max', 'simple_value': max(rewards)},
                {'tag': 'reward/min', 'simple_value': min(rewards)},
                {'tag': 'reward/hitted', 'simple_value': hitted},
                {'tag': 'reward/hitted_real', 'simple_value': hitted_real}
            ], iteration)

            s, a, discounted_r, *_ = [np.array(e) for e in zip(*trans_discounted)]
            ppo.train(s, a, discounted_r, mean_reward, iteration)


env.close()
for ppo in ppos:
    ppo.dispose()
