import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('..')
from mlagents.envs import UnityEnvironment
from dqn import DQN



train_mode = 'run' not in sys.argv
if train_mode:
    print('Training Mode')
else:
    print('Inference Mode')

env = UnityEnvironment()

default_brain_name = env.brain_names[0]

brain_params = env.brains[default_brain_name]
state_dim = brain_params.vector_observation_space_size
action_dim = brain_params.vector_action_space_size[0]
action_bound = np.array([float(i) for i in brain_params.vector_action_descriptions])




if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    with tf.Session() as sess:
        rl = DQN(
            sess=sess,
            s_dim=env.observation_space.shape[0],
            a_dim=env.action_space.n,
            batch_size=128,
            gamma=0.99,
            lr=0.01,
            epsilon=0.1,
            replace_target_iter=300
        )
        tf.global_variables_initializer().run()

        rs = []
        for i_episode in range(1000):

            s = env.reset()
            r_sum = 0
            while True:
                a = rl.choose_action(s)

                s_, r, done, _ = env.step(a)

                rl.store_transition_and_learn(s, a, r, s_, done)

                r_sum += 1
                if done:
                    print(i_episode, r_sum)
                    rs.append(r_sum)
                    break

                s = s_

        print('mean', np.mean(rs))
