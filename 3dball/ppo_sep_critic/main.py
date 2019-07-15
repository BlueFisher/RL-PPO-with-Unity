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

    Main(sys.argv[1:], Agent)
