"""
Fine tune a BERT model for sentiment classification on the IMDB data.
Will produce two models trained on half training set each, one to be used as
"defender model" and the other as "attacker model".

`frozen` is a parameter that can be used to deiced whether to fine tune the
whole model (frozen == False) or just the last layer and the classification
layer (frozen == True).
"""

import os
import gc
import time
import argparse

import torch

from subclass_avail import common
from subclass_avail.target_nlp import bert_utils


def fine_tune_bert(args):
    print('Fine tuning BERT model, received arguments:\n{}\n'.format(args))

    # Unpacking
    b_size = args['batch']
    epochs = args['epochs']
    max_len = args['length']
    n_cpu = args['workers']
    seed = args['seed']
    lr = args['learning_rate']
    frozen = args['frozen']

    # Initialization
    common.create_dirs()
    model_id = bert_utils.get_bert_name()
    device = bert_utils.get_device()  # Check if cuda available
    bert_utils.set_seed(device, seed=seed)  # Seed all the PRNGs

    exp_name_def = 'imdb_bert_{}_DEF'.format('LL' if frozen else 'FT')
    exp_name_adv = 'imdb_bert_{}_ADV'.format('LL' if frozen else 'FT')
    save_model_def = os.path.join(common.saved_models_dir, exp_name_def)
    save_model_adv = os.path.join(common.saved_models_dir, exp_name_adv)

    # Load tokenized IMDB data
    start_time = time.time()
    def_train_df_, adv_train_df, test_df = bert_utils.load_split_tokenized_data(
        dataset='imdb',
        n_cpu=n_cpu,
        max_len=max_len,
        seed=seed,
        split=True
    )
    print('Loading data took {:.2f} seconds'.format(time.time() - start_time))

    # Generate torch data loaders
    train_def_ds, train_def_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=def_train_df_,
        test_df=test_df,
        batch_size=b_size,
        shuffle=True
    )
    train_adv_ds, train_adv_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=adv_train_df,
        test_df=test_df,
        batch_size=b_size,
        shuffle=True
    )
    train_dls = {
        'DEF': (train_def_dl, save_model_def),
        'ADV': (train_adv_dl, save_model_adv)
    }

    # Fine tune BERT
    for dl_id, (train_dl, save_model) in train_dls.items():
        print('Fine tuning {} model'.format(dl_id))
        start_time = time.time()
        total_steps = len(train_dl) * epochs
        model, opt, loss, train_acc = bert_utils.train_bert(
            model_id=model_id,
            device=device,
            train_dl=train_dl,
            lr=lr,
            tot_steps=total_steps,
            epochs=epochs,
            save=save_model,
            frozen=frozen
        )
        print('Fine tuning model took {:.2f} seconds'.format(time.time() - start_time))

        # Evaluation
        pred = bert_utils.predict_bert(model, device, test_dl)
        y_test = test_df['class'].to_numpy()
        bert_utils.eval_classification(pred, y_test)

        del model, opt, loss, train_acc, pred
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='number of samples per batch', type=int, default=16)
    parser.add_argument('--epochs', help='number of epochs of fine tuning', type=int, default=4)
    parser.add_argument('--length', help='length of BERT input', type=int, default=256)
    parser.add_argument("--workers", help="number of workers to spawn", type=int, default=8)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=1e-5)
    parser.add_argument('--frozen', action='store_true', help='fine tunes only BERT last layer and classifier')
    parser.add_argument('--all', action='store_true', help='fine tunes BERT using the entire dataset')

    arguments = vars(parser.parse_args())
    fine_tune_bert(arguments)
