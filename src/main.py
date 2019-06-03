import sys
import os
import gym
from utils import parse_args

def exec_list():
    for e in gym.envs.registry.all():
        print(e.id)

def run(args):
    from learn_mnist_digits import main
    if len(args) > 1:
        command = args[1]
        if command == 'test':
            what = args[2]
            if what == 'tf':
                from test_tf import main
        elif command == 'cem':
            from cem import main
        elif command == 'ffnet':
            from ffnet import main
        elif command == 'list':
            main = exec_list
        elif command == 'rnn':
            from rnn import main
        elif command == 'gyms':
            from gyms import main
        elif command == 'learn':
            what = args[2]
            if what == 'mnist':
                from learn_mnist_digits import main
            elif what == 'xor':
                from learn_xor import main
            elif what == 'curves':
                from learn_curve_recognition import main
    main()


def ask_args(original):
    args = parse_args(input('Args: '))
    ret = original[:]
    ret.extend(args)
    return ret


if __name__ == '__main__':
    try:
        if sys.modules.get('ptvsd', None) != None:
            args = ask_args(sys.argv)
            run(args)
        else:
            run(sys.argv)
    except KeyboardInterrupt:
        pass