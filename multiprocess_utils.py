from logging import debug
from math import ceil
from multiprocessing import Pool
from numpy import concatenate
from numpy.testing import assert_array_equal
from pandas import concat, merge, read_csv
from psutil import cpu_count
from os.path import exists, join, realpath, split

from os_utils import list_files

# Directory paths.
BASE_DIR = split(realpath(__file__))[0]
TEST_DIR = join(BASE_DIR, 'test')
TEMP_DIR = join(BASE_DIR, 'temp')
TEST_FILE = join(TEST_DIR, 'test.csv')


def multiprocess_data(func, chunks, reduce=True, njobs=-1, **kwargs):
    ''' Multiprocess data.

    Parameters:
    -----------
    func: executable.
        Function to execute, should return pandas DataFrame or numpy array.
    chunks: list of tuples, (name, list of values).
    reduce: bool, default True.
        Whether to reduce the result to a single output.
    njobs: int, default -1.
        Number of processes to run. If none number of processes equals to
        cpu_count().
    kwargs: dict.
        Arguments to supply to func.

    Returns:
    --------
    result: DataFrame or List of DataFrames.
        If reduce=True then return a Data Frame, otherwise list of DataFrames.
    '''

    debug('Multiprocessing')
    if len(chunks) == 0:
        raise ValueError('Expected at least one chunk to process.')

    nchunks = set(len(chunk[1]) for chunk in chunks)
    if len(nchunks) != 1:
        raise ValueError('Chunk size is not consistent across different args.')
    nchunks = list(nchunks)[0]

    if njobs == -1:
        njobs = cpu_count()
    else:
        njobs = min(nchunks, njobs)  # we do not need more jobs than chunks
    pool = Pool(processes=njobs)
    threads = []
    for i in range(nchunks):
        for arg in chunks:
            kwargs[arg[0]] = arg[1][i]
        threads.append(pool.apply_async(func, kwds=kwargs.copy()))

    results = [p.get() for p in threads]

    debug('Multiprocessing is finished')
    if reduce:
        try:
            return concat(results, ignore_index=True)
        except:
            return concatenate(results)
    else:
        return results


def multiprocess_from_folder(func, file_arg_name, input_dir, njobs=-1, **kwargs):
    ''' Multiprocess data.

    Parameters:
    ----------
    func: executable.
        Function to execute, should return pandas DataFrame or numpy array. As
        the first argument should accept file.
    file_arg_name: string.
        The name of the first argument of func.
    input_dir: sting.
        Path to a directory where all files are saved.
    njobs: int, default -1.
        Number of processes to run. If none number of processes equals to
        cpu_count().

    Returns:
    --------
    result: DataFrame or numpy array.
    '''

    files = list_files(input_dir)
    chunks = [(file_arg_name, files)]
    return multiprocess_data(func, chunks, njobs=njobs)
