"""
Module to handle BERT fine tuning.
"""

import os
import time
import random
from multiprocessing import Pool

import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.stats import describe
from numpy import memmap, ndarray
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from subclass_avail import common
from subclass_avail.target_nlp import train_bert_attack


# SETUP

def get_bert_name():
    """ Utility to get the identifier of the BERT model to use.

    Returns:
        str: identifier of the BERT model
    """

    return 'bert-base-uncased'


def get_device():
    """ Return string identifier of available device

    Returns:
        str: 'cuda' if GPU available, else 'cpu'
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available device: ', device)

    return device


# noinspection PyUnresolvedReferences
def set_seed(device, seed):
    """ Make experiments reproducible

    Args:
        device (str): device identifier
        seed (int): PRNG seed

    Returns:

    """

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GPU deterministic mode
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# TOKENIZATION

def set_bert_separator(t):
    """ Set BERT special separator token.

    Args:
        t (list): list of tokens in texts

    Returns:

    """

    t[-1] = '[SEP]'


def preproc_bert(tokenizer, tokenized_texts_orig, max_len):
    """ Cut to max length, add special tokens, generate attention mask.

    Args:
        tokenizer (BertTokenizer): Bert Tokenizer object
        tokenized_texts_orig (list): tokenized comments
        max_len (int): maximum length of a sequence

    Returns:

    """

    common.create_dirs()

    # Cut paragraphs to max_len and set the last elements to BERT separator
    tokenized_texts = [
        ['[CLS]', ] + ts[: max_len - 1] for ts in tokenized_texts_orig
    ]
    _ = [set_bert_separator(ts) for ts in tokenized_texts]

    # Tokens must be converted to BERT's vocabulary indices before
    # passing them to the model
    x_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad up to max_len with 0s
    x_ids = [
        np.append(
            np.array(t),
            np.zeros(max_len, dtype=np.int32)
        )[:max_len] for t in x_ids
    ]

    # Compute attention mask for each vector
    attn_mask = [[float(i > 0) for i in t] for t in x_ids]

    return x_ids, attn_mask


def worker_tokenizer(tokenizer, df, max_len):
    """ Tokenization worker function

    Args:
        tokenizer (BertTokenizer): Bert Tokenizer object
        df (DataFrame): DataFrame on which to operate
        max_len (int): max length of a sequence

    Returns:
        DataFrame: tokenized and processed data
    """

    tokenized = []

    for text in df['comment_text']:
        t = tokenizer.tokenize(text)
        tokenized.append(t)

    df['tokenized_text'] = tokenized

    # If `max_len` != 0 preprocess for BERT
    if max_len:
        x_ids, attn_mask = preproc_bert(tokenizer, tokenized, max_len)
        df['text_ids'] = x_ids
        df['mask'] = attn_mask

    return df


def parallel_tokenizer(model_id, df, n_workers=4, max_len=0):
    """ Parallelize tokenizer execution

    Args:
        model_id (str): identifier of the model used
        df (DataFrame): original data
        n_workers (int): number of workers
        max_len (int): maximum length of a sequence, use 0 to compute statistics

    Returns:
        DataFrame: tokenized and processed data
    """

    # Build inputs
    split_df = np.array_split(df, n_workers)
    tokenizers = [BertTokenizer.from_pretrained(model_id)] * n_workers
    max_lens = [max_len] * n_workers
    inputs = list(zip(tokenizers, split_df, max_lens))

    pool = Pool(processes=n_workers)
    results = [
        pool.apply(worker_tokenizer, args=inputs[i]) for i in range(n_workers)
    ]

    return pd.concat(results)


def get_token_stats(train_df, model_id, n_cores=4, max_len=256):
    """ Print out statistics on the length of tokenized texts.

    Args:
        train_df (DataFrame): original data
        model_id (str): identifier of the transformer model
        n_cores (int): number of workers to spawn
        max_len (int): maximum length of a sequence

    Returns:

    """

    start_time = time.time()

    imdb_df_tk_train = parallel_tokenizer(
        model_id=model_id,
        df=train_df,
        n_workers=n_cores
    )

    print('Tokenization took {:.2f} seconds'.format(time.time() - start_time))

    tokenized_lens = [len(t) for t in imdb_df_tk_train['tokenized_text']]

    print(describe(tokenized_lens))
    print(".05 quantile: ", np.quantile(tokenized_lens, .05))
    print("Q1 quartile: ", np.quantile(tokenized_lens, .25))
    print("Q2 quartile: ", np.quantile(tokenized_lens, .50))
    print("Q3 quartile: ", np.quantile(tokenized_lens, .75))
    print(".95 quantile: ", np.quantile(tokenized_lens, .95))
    print('Number of comments under {} tokens: {}'.format(
        max_len,
        len(np.argwhere(np.array(tokenized_lens) < max_len))
    ))

    del imdb_df_tk_train


def load_split_tokenized_data(dataset='imdb', n_cpu=4, max_len=256, seed=42, split=True):
    """ Load preprocessed data. If unavailable, create it.

    Args:
        dataset (str): identifier od the dataset to load.
        n_cpu (int): number of workers to spawn
        max_len (int): maximum sequence length
        seed (int): PRNG seed
        split (bool): if False, return entire train and test set

    Returns:
        DataFrame: preprocessed data
    """

    model_id = get_bert_name()
    data_dir = common.data_dir_map[dataset]

    # Attempt loading tokenized data from stable storage
    train_tok_path = os.path.join(data_dir, 'tokenized_train.h5')
    test_tok_path = os.path.join(data_dir, 'tokenized_test.h5')

    if os.path.isfile(train_tok_path) and os.path.isfile(test_tok_path):
        train_store = pd.HDFStore(train_tok_path)
        test_store = pd.HDFStore(test_tok_path)
        train_df = train_store['df']
        test_df = test_store['df']
        train_store.close()
        test_store.close()

    else:
        print('Tokenized data not found. Generating it now.')
        data_df_path = os.path.join(data_dir, 'labeled_data.csv')

        # Check if the labeled data CSB is available
        if not os.path.isfile(data_df_path):
            print('Dataset csv file not found. Generating it now.')
            train_bert_attack.data_utils_map[dataset]()

        # Load data
        data_df = pd.read_csv(data_df_path, encoding='utf-8')
        train_df_raw = data_df[data_df['set'] == 'train']
        test_df_raw = data_df[data_df['set'] == 'test']

        # Tokenize data
        print('Tokenizing data')
        start_time = time.time()
        train_df = parallel_tokenizer(
            model_id=model_id,
            df=train_df_raw,
            n_workers=n_cpu,
            max_len=max_len
        )
        print('Tokenization took {:.2f} seconds'.format(time.time() - start_time))

        start_time = time.time()
        test_df = parallel_tokenizer(
            model_id=model_id,
            df=test_df_raw,
            n_workers=n_cpu,
            max_len=max_len
        )
        print('Tokenization took {:.2f} seconds'.format(time.time() - start_time))

        # Save to H5
        train_store = pd.HDFStore(train_tok_path)
        test_store = pd.HDFStore(test_tok_path)
        train_store['df'] = train_df
        test_store['df'] = test_df
        train_store.close()
        test_store.close()

    if not split:
        return None, train_df, test_df

    # Split train set half to defender half to adversary
    print('Splitting data sets for training.')
    train_def, train_adv = train_test_split(
        train_df,
        test_size=0.5,
        stratify=train_df['class'],
        random_state=seed,
        shuffle=True
    )

    return train_def, train_adv, test_df


# DATA LOADER MANAGEMENT

def get_torch_data_sets(train_df, test_df):
    """ Create torch data loaders.

    Args:
        train_df (DataFrame): preprocessed training data
        test_df (DataFrame):  preprocessed testing data

    Returns:
        (TensorDataset, TensorDataset): torch dataset for training and test sets
    """

    cols = ['text_ids', 'class', 'mask']

    train_x, train_y, train_att = [train_df[i].tolist() for i in cols]
    test_x, test_y, test_att = [test_df[i].tolist() for i in cols]

    print(
        'Data shapes:\n'
        'ids_train: {}\n'
        'att_train: {}\n'
        'y_train: {}\n'
        'ids_test: {}\n'
        'att_test: {}\n'
        'y_test: {}'.format(
            len(train_x),
            len(train_att),
            len(train_y),
            len(test_x),
            len(test_att),
            len(test_y)
        )
    )

    train_inputs = torch.tensor(train_x)
    train_masks = torch.tensor(train_att)
    train_labels = torch.tensor(train_y)

    test_inputs = torch.tensor(test_x)
    test_masks = torch.tensor(test_att)
    test_labels = torch.tensor(test_y)

    print(
        'Tensors shapes:\n'
        'ids_train: {}\n'
        'att_train: {}\n'
        'y_train: {}\n'
        'ids_test: {}\n'
        'att_test: {}\n'
        'y_test: {}'.format(
            train_inputs.shape,
            train_masks.shape,
            train_labels.shape,
            test_inputs.shape,
            test_masks.shape,
            test_labels.shape
        )
    )

    train_ds = TensorDataset(train_inputs, train_masks, train_labels)
    test_ds = TensorDataset(test_inputs, test_masks, test_labels)

    return train_ds, test_ds


def get_data_loaders(train_df, test_df, batch_size=8, shuffle=True):
    """ Create torch data loaders.

    Args:
        train_df (DataFrame): preprocessed training data
        test_df (DataFrame):  preprocessed testing data
        batch_size (int): size of each mini batch
        shuffle (bool): if False return a ordered train data loader

    Returns:
        (TensorDataset, DataLoader, TensorDataset, DataLoader): torch dataset
            and dataloader for training and test sets
    """

    train_ds, test_ds = get_torch_data_sets(train_df, test_df)

    train_dl = DataLoader(train_ds, shuffle=shuffle, batch_size=batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    return train_ds, train_dl, test_ds, test_dl


# BERT CORE FUNCTIONS

def wrap_train(trn_x, trn_y, trn_x_att, b_size=8, lr=1e-5, epochs=4, frozen=False, save=''):
    """ Wrapper for the training function to use in the attack.

    Args:
        trn_x (ndarray): train X array
        trn_y (ndarray): train Y array
        trn_x_att (ndarray): train attention mask array
        b_size (int): batch size
        lr (float): learning rate
        epochs (int): number of epochs to train
        frozen (bool): if true, fine tune only the last layer(s)
        save (str): path where to save the trained model

    Returns:
        BertForSequenceClassification: trained model
    """

    # Convert to torch tensors
    t_x = torch.from_numpy(trn_x)
    t_a = torch.from_numpy(trn_x_att)
    t_y = torch.from_numpy(trn_y)

    # Create torch DataSet and DataLoader
    train_ds = TensorDataset(t_x, t_a, t_y)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=b_size)

    tot_steps = len(train_dl) * epochs
    model, _, _, _ = train_bert(
        get_bert_name(),
        get_device(),
        train_dl,
        lr,
        tot_steps,
        epochs,
        save=save,
        frozen=frozen
    )

    return model


def train_bert(model_id, device, train_dl, lr, tot_steps, epochs, save='', frozen=False):
    """ Fine tune an instance of BERT.

    Args:
        model_id (str): identifier of the model
        device (str): identifier of the device
        train_dl (DataLoader): data loader for the training set
        lr (float): learning rate
        tot_steps (int): number of total steps
        epochs (int): number of epochs to train
        save (str): name of the checkpoint file to use
        frozen (bool): if true, freeze BERT and train only the classifier

    Returns:
        (BertForSequenceClassification, AdamW, list, list): trained model,
            optimizer, list of train losses, list of train accuracies
    """

    model = BertForSequenceClassification.from_pretrained(
        model_id,
        output_hidden_states=True,
        output_attentions=True,
        num_labels=2
    )
    if frozen:
        # for param in model.bert.bert.parameters():
        #     param.requires_grad = False
        for name, param in model.named_parameters():
            # if 'classifier' in name:  # classifier layer
            if 'classifier' in name or '11' in name:  # classifier and last layer
                param.requires_grad = True
            else:
                param.requires_grad = False
    model.to(device)

    # Optimizer initialization
    optimizer = AdamW(
        model.parameters(),
        lr=lr
    )

    # 0 warmup steps is default in Glue example for Huggingface Transformers
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=tot_steps
    )

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch, epochs))
        model.train()

        e_loss = 0.0
        e_steps = 0.0

        # Train the data for one epoch
        for i, batch in enumerate(tqdm.tqdm(train_dl)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Required by torch
            optimizer.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            loss = outputs[0]
            train_losses.append(loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Tracking losses
            e_loss += loss.item()
            e_steps += 1

        print("Train loss at epoch {}: {}".format(epoch, e_loss / e_steps))

        torch.cuda.empty_cache()

        # Compute train accuracy at epoch's end
        model.eval()
        e_acc = 0
        e_steps = 0

        # Evaluate data for one epoch
        for batch in train_dl:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )

            # Move logits and labels to CPU
            logits = outputs[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            e_acc += flat_accuracy(logits, label_ids)
            e_steps += 1

        print("Training accuracy - epoch {}: {}".format(epoch, e_acc / e_steps))
        train_accuracies.append(e_acc / e_steps)

        torch.cuda.empty_cache()

    # Save fine tuned model
    if not save:
        save = 'bert_tuned'
    torch.save(model.state_dict(), save + '.ckpt')

    return model, optimizer, train_losses, train_accuracies


def predict_bert(model, device, test_dl, raw=False, acc=False):
    """ Generate prediction vector.

    Args:
        model (BertForSequenceClassification): BERT model object
        device (str): device identifier
        test_dl (DataLoader): Data loader for test set
        raw (bool): if set, return also the logits
        acc (bool): if set, return global accuracy

    Returns:
        list: list of predictions (optional list of logits)
    """

    if raw and acc:
        raise Exception('raw and acc cannot be set together')

    model.to(device)
    model.eval()

    prediction = []
    prediction_raw = []
    labels = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(test_dl)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask
            )

            if raw:
                prediction_raw.append(outputs[0].tolist())

            if acc:
                labels += b_labels.tolist()

            prediction += torch.argmax(outputs[0], dim=1).tolist()

    if acc:
        return accuracy_score(labels, prediction)

    if raw:
        return prediction, prediction_raw
    return prediction


def get_representations(model_name, model, data_loader, f_name, b_size=8):
    """ Returns representations generated by trained model.

    Args:
        model_name (str): identifier of the model used
        model (BertForSequenceClassification): trained BERT instance
        data_loader (DataLoader): pytorch DataLoader instance
        f_name (str): name of the memory mapped file
        b_size (int): size of the batch

    Returns:
        memmap: memory mapped array containing representations
    """

    device = get_device()

    # Each batch representation is of size (b_size, 256, 768)
    size_buff = (b_size * len(data_loader), 256, 768)
    print('Representation size:{}'.format(size_buff))

    model.to(device)

    rep_path = common.mmap_dir
    rep_path = os.path.join(rep_path, model_name + '_' + f_name)

    if not os.path.isfile(rep_path):
        print('Representations file not found, creating it now.')
        # Create a memory mapped array
        buff = np.memmap(
            rep_path,
            dtype='float32',
            mode='w+',
            shape=size_buff
        )

        # Then we extract the representation for each batch
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(data_loader), 0):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits, hidden_states, attentions = outputs

                # Bert is 12 layers deep, the 13th layer is the classifier.
                representation = hidden_states[11].cpu().numpy()

                # Add batch representation to correct location in mmapped array
                buff[i * b_size: i * b_size + b_size] = representation

        # Flush changes to disk
        del buff

    # Each batch representation is of size (8, 256, 768)
    # Length of the training dataloader is 3125
    buff = np.memmap(
        rep_path,
        dtype='float32',
        mode='r',
        shape=size_buff
    )

    return buff


# AUXILIARY FUNCTIONS

def eval_classification(y_pred, y_true):
    """  Evaluate binary classifier.

    Args:
        y_pred (list): predicted classes
        y_true (list): true classes

    Returns:
        (dict, list): classification report and confusion matrix

    """

    cr = classification_report(y_true, y_pred, digits=6, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(classification_report(y_true, y_pred, digits=6))
    print(confusion_matrix(y_true, y_pred))

    return cm, cr


def visualize_losses(losses):
    """ Plot training loss.
    """

    plt.figure(figsize=(12, 6))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()


def visualize_accuracies(accuracies):
    """ Plot training accuracies.
    """

    plt.figure(figsize=(12, 6))
    plt.title("Training accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(accuracies)
    plt.show()


def flat_accuracy(preds, labels):
    """ Compute accuracy score.
    """

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_bert(model_file):
    """ Load trained instances of BERT

    Args:
        model_file (str): name of the checkpoint file for defender model

    Returns:
        BertForSequenceClassification: trained model

    """

    model_id = get_bert_name()

    _path = os.path.join(common.saved_models_dir, model_file)

    if not os.path.isfile(_path):
        raise FileNotFoundError('Cannot find trained BERT model: {}'.format(_path))

    print('Loading model: {}'.format(model_file))
    model = BertForSequenceClassification.from_pretrained(
        model_id,
        output_hidden_states=True,
        output_attentions=True,
        num_labels=2
    )
    model.load_state_dict(torch.load(_path))

    return model


# Attack specific

def eval_stats(model, trn_x, trn_x_att, trn_y, x_t, x_t_att, y_t,
               xt_p, xt_p_att, yt_p, b_size):
    """ Evaluate attack statistics on a BERT model.

    Args:
        model (BertForSequenceClassification): attacked model
        trn_x (ndarray): train set
        trn_x_att (ndarray): train set attention masks
        trn_y (ndarray): train set labels
        x_t (ndarray): test set
        x_t_att (ndarray): test set attention masks
        y_t (ndarray): test set labels
        xt_p (ndarray): poison set
        xt_p_att (ndarray): poison set attention masks
        yt_p (ndarray): poison set labels
        b_size (int): batch size

    Returns:
        ndarray: accuracies on train, test, and poisoned sets
    """

    device = get_device()
    accuracies = {}

    # Generate data sets and loaders
    train_ds = TensorDataset(
        torch.from_numpy(trn_x),
        torch.from_numpy(trn_x_att),
        torch.from_numpy(trn_y)
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_t),
        torch.from_numpy(x_t_att),
        torch.from_numpy(y_t)
    )
    pois_ds = TensorDataset(
        torch.from_numpy(xt_p),
        torch.from_numpy(xt_p_att),
        torch.from_numpy(yt_p)
    )

    train_dl = DataLoader(train_ds, shuffle=False, batch_size=b_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=b_size)
    pois_dl = DataLoader(pois_ds, shuffle=False, batch_size=b_size)

    all_subsets = {'train': train_dl, 'test': test_dl, 'pois': pois_dl}
    for dl_id, subset_dl in all_subsets.items():
        accuracy = predict_bert(model, device, subset_dl, acc=True)
        accuracies[dl_id] = accuracy

    return accuracies


def get_accuracy_bert(model, x, x_att, y, b_size):
    """ Compute the accuracy of a BERT model on the given data

    Args:
        model (BertForSequenceClassification): model to test
        x (ndarray): test set
        x_att (ndarray): test set attention masks
        y (ndarray): test set labels
        b_size (int): batch size

    Returns:
        float: accuracy of the model on the given data
    """

    device = get_device()

    test_ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(x_att),
        torch.from_numpy(y)
    )
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=b_size)
    accuracy = predict_bert(model, device, test_dl, acc=True)

    return accuracy
