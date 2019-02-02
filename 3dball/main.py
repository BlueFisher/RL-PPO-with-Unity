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
    'lambda': 1,
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
                                                           'agents=',
                                                           'envs=',
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
    elif opt == '--agents':
        config['agents_num'] = int(arg)
    elif opt == '--envs':
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
    cumulative_rewards_list = list()
    curr_cumulative_rewards_all = [0] * len_agents
    trans_all = [[] for _ in range(len_agents)]  # list of all transition each agent
    good_trans_all = [[] for _ in range(len_agents)]
    aux_trans_all = [[] for _ in range(len_agents)]
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
                cumulative_rewards_list.append(curr_cumulative_rewards_all[i])
                curr_cumulative_rewards_all[i] = 0

            dones[i] = dones[i] or local_dones[i]

        states = states_

    if TRAIN_MODE:
        cumulative_rewards_list.sort()
        good_cumulative_reward = cumulative_rewards_list[-int(len(cumulative_rewards_list) / 6)]

        for agents_i in range(len_agents):
            ppo = ppos[int(agents_i / config['envs_num_per_agent'])]
            trans = trans_all[agents_i]

            # compute discounted return
            v_tmp = ppo.get_v(trans[-1]['state_'][np.newaxis, :])[0]
            for tran in trans[::-1]:
                if tran['max_reached']:
                    v_tmp = ppo.get_v(tran['state_'][np.newaxis, :])[0]
                elif tran['local_done']:
                    v_tmp = 0

                v_tmp = tran['reward'] + config['gamma'] * v_tmp
                tran['discounted_return'] = v_tmp

            # compute advantage
            if config['lambda'] == 1:
                s = np.array([t['state'] for t in trans])
                v_s = ppo.get_v(s)
                for i, tran in enumerate(trans):
                    tran['advantage'] = tran['discounted_return'] - v_s[i]
            else:
                s, r, s_, done, max_reached = [np.array(e) for e in zip(*[(t['state'],
                                                                           t['reward'],
                                                                           t['state_'],
                                                                           [t['local_done']],
                                                                           [t['max_reached']]) for t in trans])]
                v_s = ppo.get_v(s)
                v_s_ = ppo.get_v(s_)
                td_errors = r + config['gamma'] * v_s_ * (~(done ^ max_reached)) - v_s
                for i, td_error in enumerate(td_errors):
                    trans[i]['td_error'] = td_error

                td_error_tmp = 0
                for tran in trans[::-1]:
                    if tran['local_done']:
                        td_error_tmp = 0

                    td_error_tmp = tran['td_error'] + config['gamma'] * config['lambda'] * td_error_tmp
                    tran['advantage'] = td_error_tmp

            # compute good trans
            is_good_tran = False
            if trans[-1]['cumulative_reward'] >= good_cumulative_reward:
                is_aux_tran = True
            else:
                is_aux_tran = False

            for tran in trans[::-1]:
                if tran['max_reached']:
                    is_good_tran = True
                    is_aux_tran = False
                elif tran['local_done']:
                    if tran['cumulative_reward'] >= good_cumulative_reward:
                        is_good_tran = False
                        is_aux_tran = True
                    else:
                        is_good_tran = is_aux_tran = False

                if is_good_tran:
                    good_trans_all[agents_i].append(tran)
                if is_aux_tran:
                    aux_trans_all[agents_i].append(tran)

        return brain_info, trans_all, rewards_all, good_trans_all, aux_trans_all
    else:
        return brain_info, None, rewards_all, None, None


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
    elif config['ppo'] == 'std':
        PPO = PPO_STD
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
                    **agent_config))

reset_config = {
    'copy': config['envs_num_per_agent'] * config['agents_num']
}

brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
for iteration in range(config['max_iter'] + 1):
    brain_info = env.reset(train_mode=TRAIN_MODE)[default_brain_name]

    brain_info, trans_all, rewards_all, good_trans_all, aux_trans_all = \
        simulate_multippo(env, brain_info, default_brain_name, action_dim, ppos)

    for ppo_i, ppo in enumerate(ppos):
        start, end = ppo_i * config['envs_num_per_agent'], (ppo_i + 1) * config['envs_num_per_agent']
        rewards = rewards_all[start:end]
        mean_reward = sum(rewards) / config['envs_num_per_agent']

        print(f'ppo {ppo_i}, iter {iteration}, rewards {", ".join([f"{i:.1f}" for i in rewards])}')

        if TRAIN_MODE:
            ppo.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': mean_reward},
                {'tag': 'reward/max', 'simple_value': max(rewards)},
                {'tag': 'reward/min', 'simple_value': min(rewards)}
            ], iteration)

            trans_for_training = list()
            for trans in trans_all[start:end]:
                trans_for_training += trans

            if config['mix']:
                good_trans = list()
                for t in good_trans_all[:start] + good_trans_all[end:]:
                    good_trans += t

                if len(good_trans) > 0:
                    s, a, discounted_r =\
                        [np.array(e) for e in zip(*[(t['state'],
                                                     t['action'],
                                                     t['discounted_return']) for t in good_trans])]
                    bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                    good_trans = [good_trans[i] for i, v in enumerate(bool_mask) if v]

                np.random.shuffle(good_trans)
                good_trans = good_trans[:int(len(trans_for_training) / 3)]

                aux_trans = list()
                for t in aux_trans_all[:start] + aux_trans_all[end:]:
                    aux_trans += t

                if len(aux_trans) > 0:
                    s, a, discounted_r =\
                        [np.array(e) for e in zip(*[(t['state'],
                                                     t['action'],
                                                     t['discounted_return']) for t in aux_trans])]
                    bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                    aux_trans = [aux_trans[i] for i, v in enumerate(bool_mask) if v]

                print(len(trans_for_training), len(good_trans), len(aux_trans))
                trans_for_training = trans_for_training + good_trans + aux_trans
                np.random.shuffle(trans_for_training)

            s, a, adv, discounted_r = [np.array(e) for e in zip(*[(t['state'],
                                                                   t['action'],
                                                                   t['advantage'],
                                                                   t['discounted_return']) for t in trans_for_training])]
            ppo.train(s, a, adv, discounted_r, iteration)

    print('=' * 20)


env.close()
for ppo in ppos:
    ppo.dispose()
