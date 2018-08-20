import gym
import sys
import os
from utils import parse_args

def main(args):
    env = gym.make('BipedalWalker-v2')
    for i_episode in range(20):
        print(f"Starting episode {i_episode}")
        observation = env.reset()
        t = 0
        action = 0
        while True:
            if t % 10 == 0:
                env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            t += 1
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break

def ask_args(original):
    args = parse_args(input('Args: '))
    ret = original[:]
    ret.extend(args)
    return ret

if __name__ == '__main__':
    try:
        if sys.modules.get('ptvsd', None) != None:
            args = ask_args(sys.argv)
            main(args)
        else:
            main(sys.argv)
    except KeyboardInterrupt:
        pass