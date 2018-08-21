import gym
import time

stop = False

def attach_close_event(window):
    if window:
        @window.event
        def on_close(*args):
            global stop
            stop = True

def run():
    global stop
    env = gym.make('BipedalWalker-v2')
    #env = gym.make('LunarLander-v2')
    framelength = 1 / 15
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
                print(observation)
                env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            t += 1
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
