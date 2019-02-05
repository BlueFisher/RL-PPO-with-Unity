import sys
import getopt
import time
import random
import os
import yaml
from functools import reduce

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
    'mix': True,
    'addition_objective': False
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

if not os.path.exists('config'):
    os.makedirs('config')
with open(f'config/{config["name"]}.yaml', 'w') as f:
    yaml.dump({**config, **agent_config}, f, default_flow_style=False)

for k, v in config.items():
    print(f'{k:>25}: {v}')
for k, v in agent_config.items():
    print(f'{k:>25}: {v}')
print('=' * 20)


class Agent(object):
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.ppo = None
        self.done = False
        self._curr_cumulative_reward = 0
        self._tmp_trans = list()
        self.trajectories = list()
        self.good_trajectories = list()
        self.aux_trajectories = list()
        self.reward = 0
        self.hitted_real = 0
        self.hitted = 0

    def add_transition(self,
                       state,
                       action,
                       reward,
                       local_done,
                       max_reached,
                       state_):
        self._curr_cumulative_reward += reward
        self._tmp_trans.append({
            'state': state,
            'action': action,
            'reward': np.array([reward]),
            'local_done': local_done,
            'max_reached': max_reached,
            'state_': state_,
            'cumulative_reward': self._curr_cumulative_reward
        })

        if not self.done:
            self.reward += reward
            if reward >= 1:
                self.hitted_real += 1
        if reward >= 1:
            self.hitted += 1

        if local_done:
            self.done = True
            self.fill_reset_tmp_trans()

    def fill_reset_tmp_trans(self):
        self.trajectories.append(self._tmp_trans)
        self._curr_cumulative_reward = 0
        self._tmp_trans = list()

    def get_cumulative_rewards(self):
        return [t[-1]['cumulative_reward'] for t in self.trajectories]

    def get_trans_combined(self):
        return [] if len(self.trajectories) == 0 else \
            reduce(lambda x, y: x + y, self.trajectories)

    def get_good_trans_combined(self):
        return [] if len(self.good_trajectories) == 0 else \
            reduce(lambda x, y: x + y, self.good_trajectories)

    def get_aux_trans_combined(self):
        return [] if len(self.aux_trajectories) == 0 else \
            reduce(lambda x, y: x + y, self.aux_trajectories)

    def compute_discounted_return(self):
        for trans in self.trajectories:
            if (not trans[-1]['max_reached']) and trans[-1]['local_done']:
                v_tmp = 0
            else:
                v_tmp = self.ppo.get_v(trans[-1]['state_'][np.newaxis, :])[0]
            for tran in trans[::-1]:
                v_tmp = tran['reward'] + config['gamma'] * v_tmp
                tran['discounted_return'] = v_tmp

    def compute_advantage(self):
        for trans in self.trajectories:
            if config['lambda'] == 1:
                s = np.array([t['state'] for t in trans])
                v_s = self.ppo.get_v(s)
                for i, tran in enumerate(trans):
                    tran['advantage'] = tran['discounted_return'] - v_s[i]
            else:
                s, r, s_, done, max_reached = [np.array(e) for e in zip(*[(t['state'],
                                                                           t['reward'],
                                                                           t['state_'],
                                                                           [t['local_done']],
                                                                           [t['max_reached']]) for t in trans])]
                v_s = self.ppo.get_v(s)
                v_s_ = self.ppo.get_v(s_)
                td_errors = r + config['gamma'] * v_s_ * (~(done ^ max_reached)) - v_s
                for i, td_error in enumerate(td_errors):
                    trans[i]['td_error'] = td_error

                td_error_tmp = 0
                for tran in trans[::-1]:
                    td_error_tmp = tran['td_error'] + config['gamma'] * config['lambda'] * td_error_tmp
                    tran['advantage'] = td_error_tmp

    def compute_good_trans(self, aux_cumulative_reward):
        for trans in self.trajectories:
            if trans[-1]['max_reached']:
                self.good_trajectories.append(trans)
            elif trans[-1]['cumulative_reward'] >= aux_cumulative_reward:
                self.aux_trajectories.append(trans)


def simulate_multippo(env, brain_info, default_brain_name, ppos: list):
    agents = [Agent(i) for i in brain_info.agents]
    for i, agent in enumerate(agents):
        ppo = ppos[int(i / config['envs_num_per_agent'])]
        agent.ppo = ppo

    states = brain_info.vector_observations
    while False in [a.done for a in agents] and not env.global_done:
        actions = []
        for i, agent in enumerate(agents):
            action = agent.ppo.choose_action(states[i][np.newaxis, :])[0]
            actions.append(action)

        brain_info = env.step({
            default_brain_name: np.array(actions)
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        max_reached = brain_info.max_reached
        states_ = brain_info.vector_observations

        for i, agent in enumerate(agents):
            agent.add_transition(states[i],
                                 actions[i],
                                 rewards[i],
                                 local_dones[i],
                                 max_reached[i],
                                 states_[i])

        states = states_

    if TRAIN_MODE:
        cumulative_rewards = list()
        for agent in agents:
            cumulative_rewards += agent.get_cumulative_rewards()
        cumulative_rewards.sort()
        aux_cumulative_reward = cumulative_rewards[-int(len(cumulative_rewards) / 6)]

        for agent in agents:
            agent.compute_discounted_return()
            agent.compute_advantage()
            if config['mix']:
                agent.compute_good_trans(aux_cumulative_reward)

    return brain_info, agents


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
        combine_ratio = 0
    elif config['ppo'] == 'std':
        PPO = PPO_STD
        combine_ratio = 1
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
                    combine_ratio=combine_ratio,
                    addition_objective=config['addition_objective'],
                    **agent_config))

reset_config = {
    'copy': config['envs_num_per_agent'] * config['agents_num']
}

brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
for iteration in range(config['max_iter'] + 1):
    brain_info = env.reset(train_mode=TRAIN_MODE)[default_brain_name]

    brain_info, agents = simulate_multippo(env, brain_info, default_brain_name, ppos)
    if TRAIN_MODE and config['mix']:
        test_states = brain_info.vector_observations
        mu = list()
        sigma = list()
        for ppo in ppos:
            _mu, _sigma = ppo.get_policy(test_states)
            mu.append(_mu.reshape(-1))
            sigma.append(_sigma.reshape(-1))

        mu_var = np.mean(np.var(mu, axis=0))
        sigma_var = np.mean(np.var(sigma, axis=0))

    for ppo_i, ppo in enumerate(ppos):
        start, end = ppo_i * config['envs_num_per_agent'], (ppo_i + 1) * config['envs_num_per_agent']
        avil_agents = agents[start:end]
        unavil_agents = agents[:start] + agents[end:]

        rewards = [a.reward for a in avil_agents]
        mean_reward = sum(rewards) / len(rewards)

        print(f'ppo {ppo_i}, iter {iteration}, rewards {", ".join([f"{i:.1f}" for i in rewards])}')

        if TRAIN_MODE:
            ppo.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': mean_reward},
                {'tag': 'reward/max', 'simple_value': max(rewards)},
                {'tag': 'reward/min', 'simple_value': min(rewards)}
            ], iteration)

            if config['mix']:
                ppo.write_constant_summaries([
                    {'tag': 'mixed_var/mu', 'simple_value': mu_var},
                    {'tag': 'mixed_var/sigma', 'simple_value': sigma_var}
                ], iteration)

            trans_for_training = list()
            for t in [a.get_trans_combined() for a in avil_agents]:
                trans_for_training += t

            if config['mix']:
                good_trans = list()
                for t in [a.get_good_trans_combined() for a in unavil_agents]:
                    good_trans += t

                if not config['addition_objective'] and len(good_trans) > 0:
                    s, a, discounted_r =\
                        [np.array(e) for e in zip(*[(t['state'],
                                                     t['action'],
                                                     t['discounted_return']) for t in good_trans])]
                    bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                    good_trans = [good_trans[i] for i, v in enumerate(bool_mask) if v]

                np.random.shuffle(good_trans)
                good_trans = good_trans[:len(trans_for_training)]

                aux_trans = list()
                for t in [a.get_aux_trans_combined() for a in unavil_agents]:
                    aux_trans += t

                if not config['addition_objective'] and len(aux_trans) > 0:
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
