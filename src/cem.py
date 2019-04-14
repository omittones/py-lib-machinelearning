from __future__ import print_function

import gym
from gym import wrappers, logger
import numpy as np
from six.moves import cPickle as pickle
import json, sys, os
from os import path
import argparse

class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, n_in, n_out):
        assert len(theta) == (n_in + 1) * n_out
        self.W = theta[0 : n_in * n_out].reshape(n_in, n_out)
        self.b = theta[n_in * n_out : None].reshape(1, n_out)
        self.n_in = n_in
        self.n_out = n_out

    def act(self, ob):
        a = ob.dot(self.W) + self.b
        return a.reshape(self.n_out)


def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

def main():

    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('target', nargs="?", default="BipedalWalker-v2")
    args = parser.parse_args()

    env = gym.make(args.target)
    env.seed(0)
    np.random.seed(0)
    params = dict(n_iter=100, batch_size=25, elite_frac = 0.2)
    num_steps = 200

    ## You provide the directory to write to (can be an existing
    ## directory, but can't contain previous monitor results. You can
    ## also dump to a tempdir if you'd like: tempfile.mkdtemp().
    #outdir = '/tmp/cem-agent-results'
    #env = wrappers.Monitor(env, outdir, force=True)

    ## Prepare snapshotting
    ## ----------------------------------------
    #def writefile(fname, s):
    #    with open(path.join(outdir, fname), 'w') as fh: fh.write(s)

    info = {}
    info['params'] = params
    info['argv'] = sys.argv
    info['env_id'] = env.spec.id
    # ------------------------------------------

    def build_policy(theta):
        nin = len(env.observation_space.low)
        nout = len(env.action_space.low)
        return ContinuousActionLinearPolicy(theta, nin, nout)

    def initial_theta():
        nin = len(env.observation_space.low)
        nout = len(env.action_space.low)
        return np.zeros(nin * nout + nout)

    def noisy_evaluation(theta):
        agent = build_policy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # Train the agent, and snapshot each stage
    theta = initial_theta()
    for (i, iterdata) in enumerate(cem(noisy_evaluation, theta, initial_std=10.0, **params)):
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        if args.display or i % 10 == 0:
            agent = build_policy(iterdata['theta_mean'])
            do_rollout(agent, env, 200, render=True)
        #writefile('agent-%.4i.pkl'%i, str(pickle.dumps(agent, -1)))
    ## Write out the env at the end so we store the parameters of this
    ## environment.
    #writefile('info.json', json.dumps(info))

    env.close()

if __name__ == '__main__':
    main()