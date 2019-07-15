from functools import reduce

import numpy as np


class Agent(object):
    reward = 0
    ppo = None

    done = False
    _curr_cumulative_reward = 0

    def __init__(self, agent_id, gamma, lambda_):
        self.agent_id = agent_id
        self.gamma = gamma
        self.lambda_ = lambda_

        self._tmp_trans = list()
        self.trajectories = list()
        self.good_trajectories = list()
        self.aux_trajectories = list()

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

        self._extra_log(state,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        state_)

        if local_done:
            self.done = True
            self.fill_reset_tmp_trans()

    def _extra_log(self,
                   state,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   state_):
        pass

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
                v_tmp = tran['reward'] + self.gamma * v_tmp
                tran['discounted_return'] = v_tmp

    def compute_advantage(self):
        for trans in self.trajectories:
            if self.lambda_ == 1:
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
                td_errors = r + self.gamma * v_s_ * (~(done ^ max_reached)) - v_s
                for i, td_error in enumerate(td_errors):
                    trans[i]['td_error'] = td_error

                td_error_tmp = 0
                for tran in trans[::-1]:
                    td_error_tmp = tran['td_error'] + self.gamma * self.lambda_ * td_error_tmp
                    tran['advantage'] = td_error_tmp

    def compute_good_trans(self, aux_cumulative_reward):
        for trans in self.trajectories:
            if trans[-1]['local_done']:
                if trans[-1]['reward'] >= 1:
                    self.good_trajectories.append(trans)
                elif trans[-1]['cumulative_reward'] >= aux_cumulative_reward:
                    self.aux_trajectories.append(trans)
