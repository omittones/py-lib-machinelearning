import gym

def main():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        print(f"Starting episode {i_episode}")
        observation = env.reset()
        t = 0
        while True:
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            t += 1
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass