import sys
import os
from utils import parse_args
from test import test
from run import run
from ffnet import ffnet
import cem

def main(args):
    if len(args) > 1:
        command = args[1]
        if command == 'test':
            test()
        elif command == 'cem':
            cem.execute()
        elif command == 'ffnet':
            ffnet.ffnet()
        elif command == 'test':
            test.test()
        else:
            run()
    else:
        run()
                
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