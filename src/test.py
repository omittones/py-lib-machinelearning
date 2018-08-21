import numpy as np
from pprint import pprint

def test():
    a = np.array([0.1,0.1,0.1,0.1]).reshape(1, 4)
    pprint(a)
    b = np.array([1,2,3,4,5,6,7,8]).reshape(4, 2)
    pprint(b)
    c = a.dot(b)
    pprint(c.reshape(2))
