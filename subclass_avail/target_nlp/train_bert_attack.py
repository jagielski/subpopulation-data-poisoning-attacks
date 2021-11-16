"""
Script to train two BERT instances and prepare for the attack.
"""

import os
import time

import torch
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from subclass_avail import common
from subclass_avail.target_nlp import bert_utils
from subclass_avail.data_utils.imdb_utils import pre_process_imdb

# Mapping to data set specific utility function
data_utils_map = {
    'imdb': pre_process_imdb
}


def load_data(dataset='imdb'):
    """ Load preprocessed data. If unavailable, create it.

    Args:
        dataset (str): identifier od the dataset to load.

    Returns:
        DataFrame: preprocessed data
    """

    data_dir = common.data_dir_map[dataset]
    data_df_path = os.path.join(data_dir, 'labeled_data.csv')

    if not os.path.isfile(data_df_path):
        print('Dataset csv file not found. Generating it now.')
        data_utils_map[dataset]()

    # Load data
    data_df = pd.read_csv(data_df_path, encoding='utf-8')

    return data_df


def train_bert(dataset='imdb', max_len=256, b_size=8, lr=1e-5, epochs=4,
               seed=42, n_cpu=8, vis=False):
    """ Fine tune BERT classifier generating attacker and defender models.

    Args:
        dataset (str): identifier of the dataset
        max_len (int): maximum sequence length
        b_size (int): size of the mini batch
        lr (float): learning rate
        epochs (int): number of epochs of training
        seed (int): PRNG seed
        n_cpu (int): number of workers to spawn
        vis (bool): if set visualize training graphs

    Returns:

    """

    print('Training BERT attacker and defender models')
    model_id = bert_utils.get_bert_name()
    # Check if cuda available
    device = bert_utils.get_device()
    common.create_dirs()

    # Load data
    data_df = load_data(dataset)
    train_df_raw = data_df[data_df['set'] == 'train']
    test_df_raw = data_df[data_df['set'] == 'test']

    # Tokenize data
    print('Tokenizing data')
    start_time = time.time()
    train_df = bert_utils.parallel_tokenizer(
        model_id=model_id,
        df=train_df_raw,
        n_workers=n_cpu,
        max_len=max_len
    )
    print('Tokenization took {:.2f} seconds'.format(time.time() - start_time))

    start_time = time.time()
    test_df = bert_utils.parallel_tokenizer(
        model_id=model_id,
        df=test_df_raw,
        n_workers=n_cpu,
        max_len=max_len
    )
    print('Tokenization took {:.2f} seconds'.format(time.time() - start_time))

    # Split train set half to defender half to adversary
    print('Splitting data sets for training.')
    train_def, train_adv = train_test_split(
        train_df,
        test_size=0.5,
        stratify=train_df['class'],
        random_state=seed,
        shuffle=True
    )

    # DEFENDER

    # Generate data loaders
    train_def_ds, train_def_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=train_def,
        test_df=test_df,
        batch_size=b_size
    )
    total_steps_def = len(train_def_dl) * epochs

    # Train the defender model
    print('Training defender model')
    def_save_path = os.path.join(common.saved_models_dir, dataset + '_bert_def')

    start_time = time.time()
    model_def, opt_def, loss_def, train_acc_def = bert_utils.train_bert(
        model_id=model_id,
        device=device,
        train_dl=train_def_dl,
        lr=lr,
        tot_steps=total_steps_def,
        epochs=epochs,
        save=def_save_path
    )
    print('Training took {:.2f} seconds'.format(time.time() - start_time))

    # Evaluate defender model
    if vis:
        bert_utils.visualize_losses(loss_def)
        bert_utils.visualize_accuracies(train_acc_def)

    def_pred = bert_utils.predict_bert(model_def, device, test_dl)
    y_test = test_df['class'].to_numpy()
    bert_utils.eval_classification(def_pred, y_test)

    # Housekeeping - avoid GPU out of memory errors
    del model_def, opt_def, loss_def, train_acc_def, train_def_dl, def_pred
    torch.cuda.empty_cache()

    # ADVERSARY

    # Generate data loaders
    train_adv_ds, train_adv_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=train_adv,
        test_df=test_df,
        batch_size=b_size
    )
    total_steps_adv = len(train_adv_dl) * epochs

    # Train the attacker model
    print('Training adversary model')
    adv_save_path = os.path.join(common.saved_models_dir, dataset + '_bert_adv')

    start_time = time.time()
    model_adv, opt_adv, loss_adv, train_acc_adv = bert_utils.train_bert(
        model_id=model_id,
        device=device,
        train_dl=train_adv_dl,
        lr=lr,
        tot_steps=total_steps_adv,
        epochs=epochs,
        save=adv_save_path
    )
    print('Training took {:.2f} seconds'.format(time.time() - start_time))

    # Evaluate adversary model
    if vis:
        bert_utils.visualize_losses(loss_adv)
        bert_utils.visualize_accuracies(train_acc_adv)

    adv_pred = bert_utils.predict_bert(model_adv, device, test_dl)
    y_test = test_df['class'].to_numpy()
    bert_utils.eval_classification(adv_pred, y_test)

    # Housekeeping - avoid GPU out of memory errors
    del model_adv, opt_adv, loss_adv, train_acc_adv, train_adv_dl
    torch.cuda.empty_cache()
