import sys
import getopt
import time
import random
import os
import yaml
from functools import reduce

import numpy as np
import tensorflow as tf

sys.path.append('../..')
from mlagents.envs import UnityEnvironment
from ppo_mixed_simple_roller import PPO

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
    'policies_num': 1,
    'agents_num_p_policy': 1,
    'seed_increment': None,
    'addition_objective': False
}
agent_config = dict()

try:
    opts, args = getopt.getopt(sys.argv[1:], 'rc:n:b:p:', ['run',
                                                           'config=',
                                                           'name=',
                                                           'build=',
                                                           'port=',
                                                           'policies=',
                                                           'agents='])
except getopt.GetoptError:
    raise Exception('ARGS ERROR')

for opt, arg in opts:
    if opt in ('-c', '--config'):
        with open(arg) as f:
            config_file = yaml.load(f)
            for k, v in config_file.items():
                if k in config.keys():
                    if k == 'build_path':
                        config['build_path'] = v[sys.platform]
                    else:
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
    elif opt == '--policies':
        config['policies_num'] = int(arg)
    elif opt == '--agents':
        config['agents_num_p_policy'] = int(arg)

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
        if len(self._tmp_trans) != 0:
            self.trajectories.append(self._tmp_trans)
            self._curr_cumulative_reward = 0
            self._tmp_trans = list()

    def get_cumulative_rewards(self):
        return [t[-1]['cumulative_reward'] for t in self.trajectories]

    def get_trans_combined(self):
        return [] if len(self.trajectories) == 0 else \
            reduce(lambda x, y: x + y, self.trajectories)

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


def simulate_multippo(env, brain_info, default_brain_name, ppo):
    agents = [Agent(i) for i in brain_info.agents]
    for i, agent in enumerate(agents):
        agent.ppo = ppo

    states = brain_info.vector_observations
    while False in [a.done for a in agents] and not env.global_done:
        actions = ppo.choose_action(states)

        brain_info = env.step({
            default_brain_name: actions
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

    # fill rest not done transitions
    for agent in agents:
        agent.fill_reset_tmp_trans()

    if TRAIN_MODE:
        for agent in agents:
            agent.compute_discounted_return()
            agent.compute_advantage()

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


print('=' * 10, config["name"], '=' * 10)
ppo = PPO(state_dim=state_dim,
          action_dim=action_dim,
          action_bound=action_bound,
          saver_model_path=f'model/{config["name"]}',
          summary_path='log' if TRAIN_MODE else None,
          summary_name=config["name"],
          seed=config['seed_increment'],
          policies_num=config['policies_num'],
          addition_objective=config['addition_objective'],
          **agent_config)

reset_config = {
    'copy': config['agents_num_p_policy'] * config['policies_num']
}

brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]
for iteration in range(config['max_iter'] + 1):
    if env.global_done:
        brain_info = env.reset(train_mode=TRAIN_MODE, config=reset_config)[default_brain_name]

    brain_info, agents = simulate_multippo(env, brain_info, default_brain_name, ppo)


    rewards = sorted([a.reward for a in agents])
    mean_reward = sum(rewards) / len(rewards)
    hitted = sum([a.hitted for a in agents])
    hitted_real = sum([a.hitted_real for a in agents])

    print(f'iter {iteration}, rewards {", ".join([f"{i:.1f}" for i in rewards])}, hitted {hitted}, hitted_real {hitted_real}')

    if TRAIN_MODE:
        ppo.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': mean_reward},
            {'tag': 'reward/max', 'simple_value': max(rewards)},
            {'tag': 'reward/min', 'simple_value': min(rewards)},
            {'tag': 'reward/hitted', 'simple_value': hitted},
            {'tag': 'reward/hitted_real', 'simple_value': hitted_real}
        ], iteration)

        trans_for_training = list()
        for t in [a.get_trans_combined() for a in agents]:
            trans_for_training += t

        s, a, adv, discounted_r = [np.array(e) for e in zip(*[(t['state'],
                                                                t['action'],
                                                                t['advantage'],
                                                                t['discounted_return']) for t in trans_for_training])]
        ppo.train(s, a, adv, discounted_r, iteration)

    print('=' * 20)


env.close()
ppo.dispose()
