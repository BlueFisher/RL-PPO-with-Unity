import sys
import logging

import numpy as np

sys.path.append('../..')
from algorithm.ppo_sep_critic_main import Main
from algorithm.agent import Agent

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - [%(name)s] - %(message)s')

    _log = logging.getLogger('tensorflow')
    _log.setLevel(logging.ERROR)

    logger = logging.getLogger('ppo')

    class AgentHitted(Agent):
        hitted = 0
        hitted_real = 0

        def _extra_log(self,
                       state,
                       action,
                       reward,
                       local_done,
                       max_reached,
                       state_):

            if not self.done and reward >= 1:
                self.hitted_real += 1
            if reward >= 1:
                self.hitted += 1

    class MainHitted(Main):
        def _log_episode_summaries(self, ppo, iteration, agents):
            rewards = np.array([a.reward for a in agents])
            hitted = sum([a.hitted for a in agents])
            hitted_real = sum([a.hitted_real for a in agents])

            ppo.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': rewards.mean()},
                {'tag': 'reward/max', 'simple_value': rewards.max()},
                {'tag': 'reward/min', 'simple_value': rewards.min()},
                {'tag': 'reward/hitted', 'simple_value': hitted},
                {'tag': 'reward/hitted_real', 'simple_value': hitted_real}
            ], iteration)

        def _log_episode_info(self, ppo_i, iteration, agents):
            rewards = [a.reward for a in agents]
            hitted = sum([a.hitted for a in agents])
            hitted_real = sum([a.hitted_real for a in agents])

            rewards_sorted = ", ".join([f"{i:.1f}" for i in sorted(rewards)])
            logger.info(f'{ppo_i}, iter {iteration}, rewards {rewards_sorted}, hitted {hitted}, hitted_real {hitted_real}')

    MainHitted(sys.argv[1:], AgentHitted)
