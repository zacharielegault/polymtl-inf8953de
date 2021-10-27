"""
Frozen lake environment
- [Documentation](https://github.com/micklethepickle/modified-frozen-lake)
- [Source](https://github.com/micklethepickle/modified-frozen-lake/blob/main/frozen_lake.py)
"""

import itertools, functools
import numpy as np
from multiprocessing import Pool, freeze_support

np.random.seed(42)


from frozen_lake import FrozenLakeEnv
from src import *
from utils import *


def main():
    env = FrozenLakeEnv(map_name="4x4-easy", slip_rate=0)
    env.seed(0)
    env.reset()

    Vs = modified_n_step_td_prediction(env, uniform_policy(env.observation_space.n, env.action_space.n), n=100)
    print("")


if __name__ == "__main__":
    freeze_support()
    main()
