"""
This script extracts the representations for IMDB data from a trained BERT model.

`frozen` is a parameter that can be used to deiced whether to use the large (FT)
model or the small (LL) model.
"""

import os
import time
import argparse

from subclass_avail import common
from subclass_avail.target_nlp import bert_utils


def representations(args):
    print('Extracting representations, received arguments:\n{}\n'.format(args))

    # Unpacking
    b_size = args['batch']
    max_len = args['length']
    n_cpu = args['workers']
    seed = args['seed']
    frozen = args['frozen']
    defender = args['defender']

    # Initializations
    device = bert_utils.get_device()  # Check if cuda available
    bert_utils.set_seed(device, seed=seed)  # Seed all the PRNGs
    model_name = 'imdb_bert_{}_{}'.format(
        'LL' if frozen else 'FT',
        'DEF' if defender else 'ADV'
    )

    # Load tokenized IMDB data
    start_time = time.time()
    train_def, train_adv, test = bert_utils.load_split_tokenized_data(
        dataset='imdb',
        n_cpu=n_cpu,
        max_len=max_len,
        seed=seed,
        split=True
    )
    print('Loading data took {:.2f} seconds'.format(time.time() - start_time))

    # Generate torch data loaders
    train_def_ds, train_def_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=train_def,
        test_df=test,
        batch_size=b_size,
        shuffle=False
    )
    train_adv_ds, train_adv_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=train_adv,
        test_df=test,
        batch_size=b_size,
        shuffle=False
    )

    # Load pre-trained BERT model
    model = bert_utils.load_bert(
        model_file=model_name + '.ckpt'
    )

    # File naming map
    file_map = {
        'll': train_def_dl,
        'll_ho': train_adv_dl,
        'll_t': test_dl
    }

    # Generate the representations
    for f_name, loader in file_map.items():
        rep_path = common.mmap_dir
        rep_path = os.path.join(rep_path, model_name + '_' + f_name)

        # If the representation files are already present remove them first
        if os.path.isfile(rep_path):
            print('{} file found, deleting it...'.format(f_name))
            os.remove(rep_path)

        # Generate the representation files and then close the memory mappings
        start_time = time.time()
        print('\nGetting {} representations'.format(f_name))
        # noinspection PyUnusedLocal
        rr = bert_utils.get_representations(
            model_name=model_name,
            model=model,
            data_loader=loader,
            f_name=f_name,
            b_size=b_size
        )
        print('Extracting representations took {:.2f} seconds'.format(time.time() - start_time))
        del rr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='number of samples per batch', type=int, default=4)
    parser.add_argument('--length', help='length of BERT input', type=int, default=256)
    parser.add_argument("--workers", help="number of workers to spawn", type=int, default=8)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument('--frozen', action='store_true', help='BERT fine tuned only on last layer and classifier')
    parser.add_argument('--defender', action='store_true', help='use defender model')

    arguments = vars(parser.parse_args())
    representations(arguments)
