import pickle


def try_load(path, raise_error=False):
    ''' Load data from the file if exists.

    Parameters:
    -----------
    path: str.
        Path to the file from which load data.
    raise_error: bool, default False.
        If True and file failed to load a value error will be raised.

    Returns:
    --------
    data: object.
        Object stored in a pickle file or None if file not or failed to read.
    '''

    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except:
        if raise_error:
            raise ValueError('%s file failed to load.' % file)
        return None


def dump_data(data, path):
    ''' Dump data to the file.

    Parameters:
    -----------
    data: object.
        Object to be stored in a pickle file.
    '''

    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
