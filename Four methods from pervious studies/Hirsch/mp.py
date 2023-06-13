# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def async_run(func, args_array, args_parser=None, desc=None, processes=min(mp.cpu_count(), 32)):
    """
    async run function

    :param func: function to run
    :param args_array: arguments array. Each item will be pass through 'func'
    :param args_parser: function to preprocess 'args' before running into 'func'
    :param desc: message displayed on the progress bar
    :param processes: number of cpu processes
    :return:
    """
    pool = mp.Pool(processes)
    result = [pool.apply_async(func, args_parser(args) if callable(args_parser) else args) for args in args_array]
    pool.close()
    result = [r.get() for r in tqdm(result, desc)]
    pool.join()
    return result
