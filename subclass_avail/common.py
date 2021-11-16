"""
General constants and auxiliary functions.
"""

import os

# PATHS
# These are the high level paths. Change to adapt to your local environment.

# Change this to local stable storage, make sure there is enough free space.
#  storage_dir = '/net/data/malware-backdoor/subpop'
storage_dir = '/media/storage/projects/research/advml/subclass'

# Directory where to store results of the attack
results_dir = 'results'

# Directories for datasets
#  imdb_data_dir = '/net/data/malware-backdoor/subpop/dataset/aclImdb'
imdb_data_dir = '/media/storage/data/imbd_reviews/aclImdb'
data_dir_map = {
    'imdb': imdb_data_dir
}

# Directory names

# Directory where to store trained/fine tuned models
saved_models_dir = os.path.join(storage_dir, 'saved_models')

# Directory containing the results of the experiments
results_dir_bert = os.path.join(results_dir, 'bert')

# Directory in which to store computed representations
#  mmap_dir = '/media/gio/storage/projects/research/advml/subclass/representations'
mmap_dir = os.path.join(storage_dir, 'representations')

ALL_DIRS = [
    storage_dir,
    results_dir,
    saved_models_dir,
    results_dir_bert,
    mmap_dir
]


# File system utilities

def create_dirs():
    """ Create directory structure.
    """

    for dir_name in ALL_DIRS:
        if not os.path.isdir(dir_name):
            print('Creating directory: {}'.format(dir_name))
            os.makedirs(dir_name)
