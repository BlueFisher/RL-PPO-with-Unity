import numpy as np
import tensorflow as tf
import sys
import time

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from util.utils import is_converged
from ppo_sep_nn import PPO

train_mode = 'run' not in sys.argv
if train_mode:
    print('Training Mode')
else:
    print('Inference Mode')

GAMMA = 0.99
ITER_MAX = 2000
AGENTS_NUM = 10


def simulate(env, brain_info):
    hitted_sum_real = 0
    hitted_sum = 0
    steps_n = 0

    len_agents = len(brain_info.agents)
    dones = [False] * len_agents
    last_states_ = [0] * len_agents
    trans_all = [[] for _ in range(len_agents)]  # 所有 agents 的 transition 集合
    rewards_sum = [0] * len_agents
    states = brain_info.vector_observations

    while False in dones and not env.global_done:
        actions = ppo.choose_action(states)

        brain_info = env.step({
            default_brain_name: actions
        })[default_brain_name]
        rewards = brain_info.rewards
        local_dones = brain_info.local_done
        max_reached = brain_info.max_reached
        states_ = brain_info.vector_observations

        for i in range(len(brain_info.agents)):
            if train_mode:
                last_states_[i] = states_[i]
                trans_all[i].append([states[i],
                                     actions[i],
                                     np.array([rewards[i]]),
                                     local_dones[i],
                                     max_reached[i]])

            if not dones[i]:
                rewards_sum[i] += rewards[i]
                if rewards[i] >= 1:
                    hitted_sum_real += 1
            if rewards[i] >= 1:
                hitted_sum += 1

            dones[i] = dones[i] or local_dones[i]
        steps_n += 1
        states = states_

    if train_mode:
        for i in range(len(brain_info.agents)):
            trans = trans_all[i]  # 每个 agent 的所有 transition
            # for j in range(len(trans) - 1, -1, -1):
            #     if trans[j][3]:
            #         break
            #     else:
            #         last_states_[i] = trans[j][0]
            v_state_ = ppo.get_v(last_states_[i])
            for tran in trans[::-1]:  # state, action, reward, done, max_reached
                if tran[3] and not tran[4]:
                    v_state_ = 0
                v_state_ = tran[2] + GAMMA * v_state_
                tran[2] = v_state_
        # 所有 agents 带 discounted_rewards 的 transition 合并为一个数组
        trans_with_discounted_rewards_all = []
        for trans in trans_all:
            trans_with_discounted_rewards_all += trans

        return brain_info, trans_with_discounted_rewards_all, rewards_sum, hitted_sum, hitted_sum_real
    else:
        return brain_info, None, rewards_sum, hitted_sum, hitted_sum_real


name = f'sep_{AGENTS_NUM}agents_3'
print(name)
reset_config = {
    'copy': AGENTS_NUM
}
env = UnityEnvironment(file_name='C:\\Users\\Fisher\\Documents\\Unity\\rl-test-build\\rl-test.exe',
                       no_graphics=True,
                       base_port=5002)
# env = UnityEnvironment()
default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = brain_params.vector_action_space_size[0]
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])

ppo = PPO(state_dim=state_dim,
          action_dim=action_dim,
          action_bound=action_bound,
          saver_model_path=f'model/{name}',
          summary_path='log' if train_mode else None,
          summary_name=name)

mean_rewards = list()
brain_info = env.reset(train_mode=train_mode, config=reset_config)[default_brain_name]
for iteration in range(ppo.init_iteration, ppo.init_iteration + ITER_MAX + 1):
    if env.global_done:
        brain_info = env.reset(train_mode=train_mode, config=reset_config)[default_brain_name]
    brain_info, trans_with_discounted_r, rewards_sum, hitted, hitted_real = simulate(env, brain_info)
    mean_reward = sum(rewards_sum) / len(rewards_sum)
    mean_rewards.append(mean_reward)

    if train_mode:
        ppo.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': mean_reward},
            {'tag': 'reward/max', 'simple_value': max(rewards_sum)},
            {'tag': 'reward/min', 'simple_value': min(rewards_sum)},
            {'tag': 'reward/hitted', 'simple_value': hitted},
            {'tag': 'reward/hitted_real', 'simple_value': hitted_real}
        ], iteration)

        if iteration % 500 == 0:
            ppo.save_model(iteration)
            # if len(mean_rewards) > 200 and is_converged(mean_rewards):
            #     print('converged')
            #     break

        s, a, discounted_r, *_ = [np.array(e) for e in zip(*trans_with_discounted_r)]
        ppo.train(s, a, discounted_r, iteration)

    print(f'iter {iteration}, rewards {mean_reward:.2f}, hitted {hitted}, hitted_real {hitted_real}')
