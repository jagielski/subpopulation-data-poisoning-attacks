"""
Module containing code to handle the IMDB classification data set.
"""

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from subclass_avail import common

# Constants
positive_class = 'pos'
negative_class = 'neg'
train_d = 'train'
test_d = 'test'

positive = 1
negative = 0
class_map = {positive_class: positive, negative_class: negative}

train_dir_pos = os.path.join(common.imdb_data_dir, train_d, positive_class)
train_dir_neg = os.path.join(common.imdb_data_dir, train_d, negative_class)
test_dir_pos = os.path.join(common.imdb_data_dir, test_d, positive_class)
test_dir_neg = os.path.join(common.imdb_data_dir, test_d, negative_class)


# PRE PROCESSING

def pre_process_imdb():
    """ Generate preprocessed DataFrame for IMDB dataset

    Returns:

    """

    print('Generating csv file for IMDB dataset.')

    start_time = time.time()

    # Setup multiprocessing - this assumes 4 workers
    n_cores = 4
    sets = ['train', 'train', 'test', 'test']
    classes = [
        class_map[positive_class],
        class_map[negative_class],
        class_map[positive_class],
        class_map[negative_class]
    ]
    file_dirs = [train_dir_pos, train_dir_neg, test_dir_pos, test_dir_neg]
    file_dirs_listed = [
        [os.path.join(i, j) for j in os.listdir(i)] for i in file_dirs
    ]

    assert len(file_dirs) == len(file_dirs_listed)

    res = pd.concat(
        [
            imdb_parallel_reader(
                file_dirs_listed[i],
                [sets[i]] * n_cores,
                [classes[i]] * n_cores,
                n_workers=n_cores
            ) for i in range(len(file_dirs_listed))
        ]
    )

    # Reset index and perform sanity check
    res = res.set_index(['id', 'set', 'class'], verify_integrity=True)

    print('Pre processing took {:.2f} seconds'.format(time.time() - start_time))

    # Save to csv
    res.to_csv(
        os.path.join(common.imdb_data_dir, 'labeled_data.csv'),
        encoding='utf-8'
    )

    return res


# HELPERS

def imdb_read_worker(dir_list, dir_set, dir_class):
    """ Worker method to read the imdb data files and clean them.

    Args:
        dir_list (list): list files to read in the directory
        dir_set (str): qualifier of the directory (train/test)
        dir_class (str): class of the points in the directory

    Returns:
        DataFrame: partial data frame with gathered data
    """

    cols = ['id', 'comment_text', 'set', 'class', 'score']
    result_df = pd.DataFrame(columns=cols)

    sets = [dir_set] * len(dir_list)
    classes = [dir_class] * len(dir_list)
    textids = []
    scores = []
    comments = []
    order_append = [textids, comments, sets, classes, scores]

    for filename in dir_list:
        textid, score = os.path.split(filename)[1].split('_')
        textid = int(textid)
        score = int(score.split('.')[0])

        with open(filename, 'r', encoding='utf-8') as infile:
            line = infile.readlines()
            assert len(line) == 1

            line = line[0].strip().replace("<br>", "")

        textids.append(textid)
        scores.append(score)
        comments.append(line)

    for i in range(len(order_append)):
        result_df[cols[i]] = order_append[i]

    return result_df


def imdb_parallel_reader(dir_list, dir_set, dir_class, n_workers=4):
    """ Spawns multiple workers to read IMDB data set files.

    Args:
        dir_list (list): list files inside data directory to split among workers
        dir_set (list): qualifiers of the directory (train/test)
        dir_class (list): classes of the points in the directory
        n_workers (int): number of workers to spawn

    Returns:
        DataFrame: data frame containing the data for specified class/set
    """

    dir_lists = np.array_split(dir_list, n_workers)

    inputs = list(zip(dir_lists, dir_set, dir_class))
    assert len(inputs) == n_workers

    pool = Pool(processes=n_workers)
    results = [
        pool.apply(imdb_read_worker, args=inputs[i]) for i in range(n_workers)
    ]

    return pd.concat(results)
