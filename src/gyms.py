import gym
import time

stop = False

def attach_close_event(window):
    if window:
        @window.event
        def on_close(*args): #pylint: disable=unused-variable
            global stop
            stop = True

def main():
    global stop
    env = gym.make('MountainCar-v0')
    framelength = 1.0 / 60
    for i_episode in range(20):
        print(f"Starting episode {i_episode}")
        t = 0
        last = time.time()
        observation = env.reset()
        env.render()
        attach_close_event(env.unwrapped.viewer.window)
        while not stop:
            now = time.time()
            if now > last + framelength:
                last = now
                env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            t += 1
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break

if __name__ == '__main__':
    main()