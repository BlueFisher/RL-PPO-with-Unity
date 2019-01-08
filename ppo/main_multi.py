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
ITER_MAX = 1000
SAVE_PER_ITER = 200
PPO_NUM = 5
AGENTS_NUM_EACH_PPO = 4


def simulate_multippo(env, brain_info, default_brain_name, action_dim, ppos: list):
    steps_n = 0

    len_agents = len(brain_info.agents)
    dones = [False] * len_agents
    last_states_ = [0] * len_agents
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
            if train_mode:
                last_states_[i] = states_[i]
                trans_all[i].append([states[i],
                                     actions[i],
                                     np.array([rewards[i]]),
                                     local_dones[i],
                                     max_reached[i]])

            if not dones[i]:
                rewards_all[i] += rewards[i]
                if rewards[i] >= 1:
                    hitted_real_all[i] += 1
            if rewards[i] >= 1:
                hitted_all[i] += 1

            dones[i] = dones[i] or local_dones[i]
        steps_n += 1
        states = states_

    if train_mode:
        for i in range(len_agents):
            trans = trans_all[i]  # all transitions in each agent
            # for j in range(len(trans) - 1, -1, -1):
            #     if trans[j][3]:
            #         break
            #     else:
            #         last_states_[i] = trans[j][0]

            v_state_ = ppos[int(i / AGENTS_NUM_EACH_PPO)].get_v(last_states_[i])
            for tran in trans[::-1]:  # state, action, reward, done, max_reached
                if tran[3]:
                    v_state_ = 0
                v_state_ = tran[2] + GAMMA * v_state_
                tran[2] = v_state_

        trans_with_discounted_rewards_all_ppos = []
        for i in range(len(ppos)):
            # all transitions with discounted rewards of each agent merge into one list
            trans_with_discounted_rewards = []
            for trans in trans_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO]:
                trans_with_discounted_rewards += trans
            trans_with_discounted_rewards_all_ppos.append(trans_with_discounted_rewards)

        return brain_info, trans_with_discounted_rewards_all_ppos, rewards_all, hitted_all, hitted_real_all
    else:
        return brain_info, None, rewards_all, hitted_all, hitted_real_all


env = UnityEnvironment(file_name='C:\\Users\\Fisher\\Documents\\Unity\\rl-test-build\\rl-test.exe',
                       no_graphics=True,
                       base_port=7000)
# env = UnityEnvironment()
default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = brain_params.vector_action_space_size[0]
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])

ppos = []
for i in range(PPO_NUM):
    name = f'sep_{AGENTS_NUM_EACH_PPO}agents_nobound/{i+1}'
    print(name)
    ppos.append(PPO(state_dim=state_dim,
                    action_dim=action_dim,
                    action_bound=action_bound,
                    saver_model_path=f'model/{name}',
                    summary_path='log' if train_mode else None,
                    summary_name=name))


reset_config = {
    'copy': AGENTS_NUM_EACH_PPO * PPO_NUM
}

# mean_rewards = list()
brain_info = env.reset(train_mode=train_mode, config=reset_config)[default_brain_name]
for iteration in range(ITER_MAX + 1):
    if env.global_done:
        brain_info = env.reset(train_mode=train_mode, config=reset_config)[default_brain_name]
    brain_info, trans_with_discounted_rewards_all_ppos, \
        rewards_all, hitted_all, hitted_real_all = simulate_multippo(env,
                                                                     brain_info,
                                                                     default_brain_name,
                                                                     action_dim,
                                                                     ppos)

    # mean_rewards.append(mean_reward)

    for i, ppo in enumerate(ppos):
        rewards = rewards_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO]
        mean_reward = sum(rewards) / AGENTS_NUM_EACH_PPO
        hitted = sum(hitted_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO])
        hitted_real = sum(hitted_real_all[i * AGENTS_NUM_EACH_PPO:(i + 1) * AGENTS_NUM_EACH_PPO])

        if train_mode:
            ppo.write_constant_summaries([
                {'tag': 'reward/mean', 'simple_value': mean_reward},
                {'tag': 'reward/max', 'simple_value': max(rewards)},
                {'tag': 'reward/min', 'simple_value': min(rewards)},
                {'tag': 'reward/hitted', 'simple_value': hitted},
                {'tag': 'reward/hitted_real', 'simple_value': hitted_real}
            ], iteration)

            if iteration % SAVE_PER_ITER == 0:
                ppo.save_model(iteration)
                # if len(mean_rewards) > 200 and is_converged(mean_rewards):
                #     print('converged')
                #     break

            s, a, discounted_r, *_ = [np.array(e) for e in
                                      zip(*trans_with_discounted_rewards_all_ppos[i])]
            ppo.train(s, a, discounted_r, iteration)

        print(f'ppo {i+1}, iter {iteration}, rewards {mean_reward:.2f}, hitted {hitted}, hitted_real {hitted_real}')

env.close()
for ppo in ppos:
    ppo.dispose()
