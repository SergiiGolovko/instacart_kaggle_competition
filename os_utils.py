from os import listdir
from os.path import isfile, join


def list_files(path):
    ''' List all files in the directory.
    '''

    file_names = listdir(path)
    files = [join(path, f) for f in file_names if isfile(join(path, f))]
    return files
