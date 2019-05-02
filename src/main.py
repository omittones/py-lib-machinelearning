import sys
import os
import gym
from utils import parse_args

def main(args):
    if len(args) > 1:
        command = args[1]
        if command == 'test':
            from test import main
            main()
        elif command == 'cem':
            from cem import main
            main()
        elif command == 'ffnet':
            from ffnet import main
            main()
        elif command == 'list':
            for e in gym.envs.registry.all():
                print(e.id)
        elif command == 'rnn':
            from rnn import main
            main()
        else:
            from gyms import main
            main()
    else:
        from gyms import main
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
            main(args)
        else:
            main(sys.argv)
    except KeyboardInterrupt:
        pass