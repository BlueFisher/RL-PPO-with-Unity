from collections import deque
from pathlib import Path
import functools
import getopt
import importlib
import logging
import os
import sys
import time
import yaml

import numpy as np
import tensorflow as tf

from .ppo_sep_critic_base import Critic_Base, PPO_Base
from .agent import Agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlagents.envs import UnityEnvironment

logger = logging.getLogger('ppo')


class Main(object):
    train_mode = True

    def __init__(self, argv, agent_class=Agent):
        self._now = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self._agent_class = agent_class

        self.config, self.reset_config, critic_config, ppo_config, model_root_path = self._init_config(argv)
        self._init_env(critic_config, ppo_config, model_root_path)
        self._run()

    def _init_config(self, argv):
        config = {
            'name': self._now,
            'build_path': None,
            'scene': None,
            'port': 7000,
            'ppo': 'ppo',
            'lambda': 1,
            'gamma': 0.99,
            'max_iter': 2000,
            'policies_num': 1,
            'agents_num_p_policy': 1,
            'reset_on_iteration': True,
            'seed': None,
            'mix': True,
            'aux_cumulative_ratio': 0.4,
            'good_trans_ratio': 1,
            'addition_objective': False
        }
        reset_config = {
            'copy': 1
        }
        critic_config = dict()
        ppo_config = dict()

        try:
            opts, args = getopt.getopt(argv, 'rc:n:b:p:', ['run',
                                                           'config=',
                                                           'name=',
                                                           'build=',
                                                           'port=',
                                                           'seed=',
                                                           'ppo=',
                                                           'policies=',
                                                           'agents=',
                                                           'no_mix'])
        except getopt.GetoptError:
            raise Exception('ARGS ERROR')

        for opt, arg in opts:
            if opt in ('-c', '--config'):
                with open(arg) as f:
                    config_file = yaml.load(f, Loader=yaml.FullLoader)
                    for k, v in config_file.items():
                        if k == 'build_path':
                            config['build_path'] = v[sys.platform]

                        elif k == 'reset_config':
                            reset_config = dict(reset_config, **({} if v is None else v))

                        elif k == 'ppo_config':
                            ppo_config = {} if v is None else v

                        elif k == 'critic_config':
                            critic_config = {} if v is None else v

                        else:
                            config[k] = v
                break

        for opt, arg in opts:
            if opt in ('-r', '--run'):
                self.train_mode = False
            elif opt in ('-n', '--name'):
                config['name'] = arg.replace('{time}', self._now)
            elif opt in ('-b', '--build'):
                config['build_path'] = arg
            elif opt in ('-p', '--port'):
                config['port'] = int(arg)
            elif opt == '--seed':
                ppo_config['seed'] = int(arg)
            elif opt == '--ppo':
                config['ppo'] = arg
            elif opt == '--policies':
                config['policies_num'] = int(arg)
            elif opt == '--agents':
                config['agents_num_p_policy'] = int(arg)

            elif opt == '--no_mix':
                config['mix'] = False

        reset_config['copy'] = config['policies_num'] * config['agents_num_p_policy']

        model_root_path = f'models/{config["name"]}'

        if self.train_mode:
            if not os.path.exists(model_root_path):
                os.makedirs(model_root_path)
            with open(f'{model_root_path}/config.yaml', 'w') as f:
                yaml.dump({**config,
                           'critic_config': {**critic_config},
                           'ppo_config': {**ppo_config}
                           },
                          f, default_flow_style=False)

        config_str = '\ncommon_config'
        for k, v in config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nreset_config:'
        for k, v in reset_config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\n\critic_config:'
        for k, v in critic_config.items():
            config_str += f'\n{k:>25}: {v}'

        config_str += '\nppo_config:'
        for k, v in ppo_config.items():
            config_str += f'\n{k:>25}: {v}'

        logger.info(config_str)

        return config, reset_config, critic_config, ppo_config, model_root_path

    def _init_env(self, critic_config, ppo_config, model_root_path):
        config = self.config

        if config['build_path'] is None or config['build_path'] == '':
            self.env = UnityEnvironment()
        else:
            self.env = UnityEnvironment(file_name=config['build_path'],
                                        no_graphics=self.train_mode,
                                        base_port=config['port'],
                                        args=['--scene', config['scene']])

        self.default_brain_name = self.env.brain_names[0]

        brain_params = self.env.brains[self.default_brain_name]
        state_dim = brain_params.vector_observation_space_size
        action_dim = brain_params.vector_action_space_size[0]

        ppo_module = importlib.import_module(config['ppo'])

        class Critic(ppo_module.Critic_Custom, Critic_Base):
            pass

        class PPO(ppo_module.PPO_Custom, PPO_Base):
            pass

        self.critic = Critic(state_dim=state_dim,
                             model_root_path=model_root_path,
                             seed=config['seed'],
                             **critic_config)

        self.ppos = list()

        for i in range(config['policies_num']):
            if config['policies_num'] > 1:
                tmp_model_root_path = f'{model_root_path}/{i}'
            else:
                tmp_model_root_path = model_root_path

            if config['seed'] is None:
                seed = None
            else:
                seed = i + self.config['seed']

            logger.info(tmp_model_root_path)
            ppo = PPO(state_dim=state_dim,
                      action_dim=action_dim,
                      model_root_path=tmp_model_root_path,
                      seed=seed,
                      addition_objective=config['addition_objective'],
                      **ppo_config)
            ppo.get_v = lambda s: self.critic.get_v(s)
            self.ppos.append(ppo)

    def _simulate_multippo(self, env, brain_info, default_brain_name):
        agents = [self._agent_class(i, self.config['gamma'], self.config['lambda'])
                  for i in brain_info.agents]
        for i, agent in enumerate(agents):
            ppo = self.ppos[int(i / self.config['agents_num_p_policy'])]
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
        # # fill rest not done transitions
        # for agent in agents:
        #     agent.fill_reset_tmp_trans()

        if self.train_mode:
            cumulative_rewards = list()
            for agent in agents:
                cumulative_rewards += agent.get_cumulative_rewards()
            cumulative_rewards.sort()
            aux_cumulative_reward = cumulative_rewards[-int(len(cumulative_rewards) * self.config['aux_cumulative_ratio'])]

            for agent in agents:
                agent.compute_discounted_return()
                agent.compute_advantage()
                if self.config['mix']:
                    agent.compute_good_trans(aux_cumulative_reward)

        return brain_info, agents

    def _run(self):
        config = self.config

        brain_info = self.env.reset(train_mode=self.train_mode, config=self.reset_config)[self.default_brain_name]

        for iteration in range(config['max_iter'] + 1):
            if self.env.global_done or config['reset_on_iteration']:
                brain_info = self.env.reset(train_mode=self.train_mode)[self.default_brain_name]

            brain_info, agents = self._simulate_multippo(self.env, brain_info, self.default_brain_name)

            trans_for_critic_training = list()
            for ppo_i, ppo in enumerate(self.ppos):
                start, end = ppo_i * config['agents_num_p_policy'], (ppo_i + 1) * config['agents_num_p_policy']
                avail_agents = agents[start:end]
                unavail_agents = agents[:start] + agents[end:]

                rewards = sorted([a.reward for a in avail_agents])
                mean_reward = sum(rewards) / len(rewards)

                self._log_episode_info(ppo_i, iteration, avail_agents)

                if self.train_mode:
                    self._log_episode_summaries(ppo, iteration, avail_agents)

                    trans_for_training = list()
                    for t in [a.get_trans_combined() for a in avail_agents]:
                        trans_for_training += t
                        trans_for_critic_training += t

                    if config['mix']:
                        good_trans = list()
                        for t in [a.get_good_trans_combined() for a in unavail_agents]:
                            good_trans += t
                            trans_for_critic_training += t

                        if not config['addition_objective'] and len(good_trans) > 0:
                            s, a = [np.array(e) for e in zip(*[(t['state'],
                                                                t['action']) for t in good_trans])]
                            bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                            good_trans = [good_trans[i] for i, v in enumerate(bool_mask) if v]

                        np.random.shuffle(good_trans)
                        good_trans = good_trans[:int(len(trans_for_training) * config['good_trans_ratio'])]

                        aux_trans = list()
                        for t in [a.get_aux_trans_combined() for a in unavail_agents]:
                            aux_trans += t

                        if not config['addition_objective'] and len(aux_trans) > 0:
                            s, a = [np.array(e) for e in zip(*[(t['state'],
                                                                t['action']) for t in aux_trans])]
                            bool_mask = ppo.get_not_zero_prob_bool_mask(s, a)
                            aux_trans = [aux_trans[i] for i, v in enumerate(bool_mask) if v]

                        trans_for_training = trans_for_training + good_trans + aux_trans
                        np.random.shuffle(trans_for_training)

                    s, a, adv = [np.array(e) for e in zip(*[(t['state'],
                                                            t['action'],
                                                            t['advantage']) for t in trans_for_training])]
                    ppo.train(s, a, adv, iteration)

            np.random.shuffle(trans_for_critic_training)
            s, discounted_r = [np.array(e) for e in zip(*[(t['state'],
                                                           t['discounted_return']) for t in trans_for_critic_training])]
            self.critic.train(s, discounted_r, iteration)

            logger.info('=' * 20)

        self.env.close()

    def _log_episode_summaries(self, ppo, iteration, agents):
        rewards = np.array([a.reward for a in agents])
        ppo.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ], iteration)

    def _log_episode_info(self, ppo_i, iteration, agents):
        rewards = [a.reward for a in agents]
        rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
        logger.info(f'{ppo_i}, iter {iteration}, rewards {rewards_sorted}')
