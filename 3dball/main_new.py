import sys
import getopt
import time
import random
import yaml

import numpy as np
import tensorflow as tf

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from ppo_3dball import PPO_SEP, PPO_STD

NOW = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
TRAIN_MODE = True

config = {
    'name': NOW,
    'build_path': None,
    'port': 7000,
    'ppo': 'sep',
    'gamma': 0.99,
    'max_iter': 2000,
    'agents_num': 1,
    'envs_num_per_agent': 1,
    'seed_increment': None,
    'mix': True
}
agent_config = dict()

try:
    opts, args = getopt.getopt(sys.argv[1:], 'rc:n:b:p:', ['run',
                                                           'config=',
                                                           'name=',
                                                           'build=',
                                                           'port=',
                                                           'ppo=',
                                                           'agents_num=',
                                                           'envs_num=',
                                                           'no_mix'])
except getopt.GetoptError:
    raise Exception('ARGS ERROR')

for opt, arg in opts:
    if opt in ('-c', '--config'):
        with open(arg) as f:
            config_file = yaml.load(f)
            for k, v in config_file.items():
                if k in config.keys():
                    config[k] = v
                else:
                    agent_config[k] = v
        break

for opt, arg in opts:
    if opt in ('-r', '--run'):
        TRAIN_MODE = False
    elif opt in ('-n', '--name'):
        config['name'] = arg.replace('{time}', NOW)
    elif opt in ('-b', '--build'):
        config['build_path'] = arg
    elif opt in ('-p', '--port'):
        config['port'] = int(arg)
    elif opt == '--ppo':
        config['ppo'] = arg
    elif opt == '--agents_num':
        config['agents_num'] = int(arg)
    elif opt == '--envs_num':
        config['envs_num_per_agent'] = int(arg)

    elif opt == '--no_mix':
        config['mix'] = False


for k, v in config.items():
    print(f'{k:>25}: {v}')
for k, v in agent_config.items():
    print(f'{k:>25}: {v}')
print('=' * 20)


def simulate_multippo(env, brain_info, default_brain_name, action_dim, ppos: list):
    len_agents = len(brain_info.agents)
    dones = [False] * len_agents
    cumulative_rewards = list()
    curr_cumulative_rewards_all = [0] * len_agents
    trans_all = [[] for _ in range(len_agents)]  # list of all transition each agent
    rewards_all = [0] * len_agents

    states = brain_info.vector_observations
    while False in dones and not env.global_done:
        actions = np.zeros((len_agents, action_dim))
        for i, ppo in enumerate(ppos):
            actions[i * config['envs_num_per_agent']:(i + 1) * config['envs_num_per_agent']] \
                = ppo.choose_action(states[i * config['envs_num_per_agent']:(i + 1) * config['envs_num_per_agent']])

        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        max_reached = brain_info.max_reached
        states_ = brain_info.vector_observations

        for i in range(len_agents):
            if TRAIN_MODE:
                curr_cumulative_rewards_all[i] += rewards[i]
                trans_all[i].append({
                    'state': states[i],
                    'action': actions[i],
                    'reward': np.array([rewards[i]]),
                    'local_done': local_dones[i],
                    'max_reached': max_reached[i],
                    'state_': states_[i],
                    'cumulative_reward': curr_cumulative_rewards_all[i]
                })

            if not dones[i]:
                rewards_all[i] += rewards[i]

            if local_dones[i]:
                cumulative_rewards.append(curr_cumulative_rewards_all[i])
                curr_cumulative_rewards_all[i] = 0

            dones[i] = dones[i] or local_dones[i]

        states = states_

    return brain_info, trans_all, rewards_all, sorted(cumulative_rewards)


if config['build_path'] is None or config['build_path'] == '':
    env = UnityEnvironment()
else:
    env = UnityEnvironment(file_name=config['build_path'],
                           no_graphics=TRAIN_MODE,
                           base_port=config['port'])

# single brain
default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = brain_params.vector_action_space_size[0]
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])

ppos = []
for i in range(config['agents_num']):
    if config['agents_num'] > 1:
        name = f'{config["name"]}/{i}'
    else:
        name = config['name']

    if config['seed_increment'] is None:
        seed = None
    else:
        seed = i + config['seed_increment']

    if config['ppo'] == 'sep':
        PPO = PPO_SEP
        combine_loss = False
    elif config['ppo'] == 'std':
        PPO = PPO_STD
        combine_loss = True
    else:
        raise Exception(f'PPO name {config["ppo"]} is in correct')

    print('=' * 10, name, '=' * 10)
    ppos.append(PPO(state_dim=state_dim,
                    action_dim=action_dim,
                    action_bound=action_bound,
                    saver_model_path=f'model/{name}',
                    summary_path='log' if TRAIN_MODE else None,
                    summary_name=name,
                    seed=seed,
                    combine_loss=combine_loss,
                    **agent_config))


def get_v_average(state):
    vs = [ppo.get_v(state) for ppo in ppos]
    return sum(vs) / len(vs)


reset_config = {
    'copy': config['envs_num_per_agent'] * config['agents_num']
}

brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
for iteration in range(config['max_iter'] + 1):
    brain_info = env.reset(train_mode=TRAIN_MODE)[default_brain_name]

    brain_info, trans_all, rewards_all, cumulative_rewards = \
        simulate_multippo(env, brain_info, default_brain_name, action_dim, ppos)

    good_cumulative_reward = cumulative_rewards[-int(len(cumulative_rewards) / 6)]

    for i in range(len(trans_all)):
        for tran in trans_all[i]:
            tran['discounted_return'] = 0

    for i, ppo in enumerate(ppos):
        start, end = i * config['envs_num_per_agent'], (i + 1) * config['envs_num_per_agent']
        trans_for_training = []
        trans_not_self_for_training = []
        trans_auxiliary_for_traning = []
        for j in range(len(trans_all)):
            trans = trans_all[j]

            if trans[-1]['cumulative_reward'] >= good_cumulative_reward:
                is_good_tran_not_terminal = True
            else:
                is_good_tran_not_terminal = False
            is_good_tran = False

            v_state_ = ppo.get_v(trans[-1]['state_'])

            for tran in trans[::-1]:
                if tran['max_reached']:
                    v_state_ = ppo.get_v(tran['state_'])
                    is_good_tran_not_terminal = True
                    is_good_tran = False
                elif tran['local_done']:  # not max_reached but done
                    v_state_ = 0
                    is_good_tran_not_terminal = False
                    if tran['cumulative_reward'] >= good_cumulative_reward:
                        is_good_tran = True
                    else:
                        is_good_tran = False
                v_state_ = tran['reward'] + config['gamma'] * v_state_
                tran['discounted_return'] = v_state_

                if start <= j < end:
                    trans_for_training.append(tran)
                else:
                    if is_good_tran:
                        trans_not_self_for_training.append(tran)
                    elif is_good_tran_not_terminal:
                        trans_auxiliary_for_traning.append(tran)

        rewards = rewards_all[start:end]
        mean_reward = sum(rewards) / config['envs_num_per_agent']

        print(f'ppo {i}, iter {iteration}, rewards {", ".join([f"{i:.1f}" for i in rewards])}')

        if TRAIN_MODE:
            ppo.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': mean_reward},
                {'tag': 'reward/max', 'simple_value': max(rewards)},
                {'tag': 'reward/min', 'simple_value': min(rewards)}
            ], iteration)

            if config['mix']:
                trans_not_self_for_training_filtered = []
                if len(trans_not_self_for_training) > 0:
                    s, a, discounted_r =\
                        [np.array(e) for e in zip(*[(t['state'],
                                                     t['action'],
                                                     t['discounted_return']) for t in trans_not_self_for_training])]
                    bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                    # importance = np.squeeze(ppo.get_importance(s, discounted_r))
                    for j, tran in enumerate(trans_not_self_for_training):
                        if bool_mask[j]:
                            trans_not_self_for_training_filtered.append(tran)

                trans_auxiliary_for_traning_filtered = []
                if len(trans_auxiliary_for_traning) > 0:
                    s, a, discounted_r =\
                        [np.array(e) for e in zip(*[(t['state'],
                                                     t['action'],
                                                     t['discounted_return']) for t in trans_auxiliary_for_traning])]
                    bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                    # importance = np.squeeze(ppo.get_importance(s, discounted_r))
                    for j, tran in enumerate(trans_auxiliary_for_traning):
                        if bool_mask[j]:
                            trans_auxiliary_for_traning_filtered.append(tran)

                np.random.shuffle(trans_auxiliary_for_traning_filtered)
                trans_auxiliary_for_traning_filtered = trans_auxiliary_for_traning_filtered[:int(len(trans_auxiliary_for_traning_filtered) / 4)]

                trans_for_training = trans_for_training + trans_not_self_for_training_filtered + trans_auxiliary_for_traning_filtered

                np.random.shuffle(trans_for_training)

            s, a, discounted_r = \
                [np.array(e) for e in zip(*[(t['state'],
                                             t['action'],
                                             t['discounted_return']) for t in trans_for_training])]
            ppo.train(s, a, discounted_r, iteration)

    print('=' * 20)


env.close()
for ppo in ppos:
    ppo.dispose()
