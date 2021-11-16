import os
import gc
import argparse

import torch
import numpy as np

from subclass_avail import common
from subclass_avail import attack_utils
from subclass_avail.target_nlp import bert_utils


def init_cluster_attack(frozen, n_clusters, pca_dim):
    """ Initialize the attack.

    Splits the data in 3 distinct sets:
    - adversary data
    - defender data
    - test data

    For each of them return:
    - the data
    - original labels
    - learned representation
    - clustering result
    - logits output

    Returns:
        tuple: artifacts for each data set
    """

    # Get device id
    device = bert_utils.get_device()
    batch = 4  # This is fixed to avoid errors in the representation loading

    # Load pre-trained adversary BERT model
    model_name_adv = 'imdb_bert_{}_ADV'.format('LL' if frozen else 'FT')
    model_adv = bert_utils.load_bert(model_file=model_name_adv + '.ckpt')

    # Load dataset and split it into adversary and defender sets
    train_def, train_adv, test = bert_utils.load_split_tokenized_data()
    train_def_ds, train_def_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=train_def,
        test_df=test,
        batch_size=batch,
        shuffle=False
    )
    train_adv_ds, train_adv_dl, test_ds, test_dl = bert_utils.get_data_loaders(
        train_df=train_adv,
        test_df=test,
        batch_size=batch,
        shuffle=False
    )

    # Use the adversary's model to extract representations and logits
    model_adv.eval()
    print('\nGetting def train representations')
    ll = bert_utils.get_representations(
        model_name=model_name_adv,
        model=model_adv,
        data_loader=train_def_dl,
        f_name='ll',
        b_size=batch
    )
    print('\nGetting adv train representations')
    ll_ho = bert_utils.get_representations(
        model_name=model_name_adv,
        model=model_adv,
        data_loader=train_adv_dl,
        f_name='ll_ho',
        b_size=batch
    )
    print('\nGetting test representations')
    ll_t = bert_utils.get_representations(
        model_name=model_name_adv,
        model=model_adv,
        data_loader=test_dl,
        f_name='ll_t',
        b_size=batch
    )

    # Compute predictions on both training sets with the adversary model
    print('\nComputing predictions on the training sets')
    preds_raw = bert_utils.predict_bert(
        model=model_adv,
        device=device,
        test_dl=train_def_dl,
        raw=True
    )
    preds_ho_raw = bert_utils.predict_bert(
        model=model_adv,
        device=device,
        test_dl=train_adv_dl,
        raw=True
    )
    preds = np.concatenate(preds_raw[1])
    preds_ho = np.concatenate(preds_ho_raw[1])

    ll = ll.reshape((ll.shape[0], -1))
    ll_ho = ll_ho.reshape((ll_ho.shape[0], -1))
    ll_t = ll_t.reshape((ll_t.shape[0], -1))
    print('\nShapes\n\tll: {}\n\tll_ho: {}\n\tll_t: {}'.format(ll.shape, ll_ho.shape, ll_t.shape))

    print('\nClustering ll_ho')
    labels_ho, cl_fn = attack_utils.clustering(
        ll_ho,
        clusters=n_clusters,
        pca_dim=pca_dim
    )
    print('\nClustering ll_t')
    labels_t = cl_fn(ll_t)
    print('\nClustering ll')
    labels = cl_fn(ll)

    # extract numpy arrays from pytorch DataSet objects
    x = (train_def_ds.tensors[0].numpy(), train_def_ds.tensors[1].numpy())
    y = train_def_ds.tensors[2].numpy()
    x_ho = (train_adv_ds.tensors[0].numpy(), train_adv_ds.tensors[1].numpy())
    y_ho = train_adv_ds.tensors[2].numpy()
    x_t = (test_ds.tensors[0].numpy(), test_ds.tensors[1].numpy())
    y_t = test_ds.tensors[2].numpy()

    del model_adv
    gc.collect()
    torch.cuda.empty_cache()

    return (x, y, ll, labels, preds), \
           (x_ho, y_ho, ll_ho, labels_ho, preds_ho), \
           (x_t, y_t, ll_t, labels_t)


def attack(args):
    print('Performing sub-population poisoning attack, received arguments:\n{}\n'.format(args))

    # Unpack
    n_clusters = args['n_clusters']
    pca_dim = args['pca_dim']
    seed = args['seed']
    frozen = args['frozen']
    pois_rate = args['poison_rate']
    cl_ind = args['class_ind']

    # Initialization
    device = bert_utils.get_device()  # Check if cuda available
    bert_utils.set_seed(device, seed=seed)  # Seed all the PRNGs

    # Initialize attack
    (x, y, ll, labels, preds), (x_ho, y_ho, ll_ho, labels_ho, preds_ho), (
        x_t, y_t, ll_t, labels_t) = init_cluster_attack(
        frozen=frozen,
        n_clusters=n_clusters,
        pca_dim=pca_dim
    )

    x, x_att = x
    x_ho, x_ho_att = x_ho
    x_t, x_t_att = x_t

    l_d = np.unique(labels, return_counts=True)
    lt_d = np.unique(labels_t, return_counts=True)
    lho_d = np.unique(labels_ho, return_counts=True)

    print("labels distr", l_d)
    print("ho labels distr", lho_d)
    print("test distr", lt_d)
    print('\nx shape: {}\nx_ho shape:{}\nx_t shape: {}'.format(x.shape, x_ho.shape, x_t.shape))

    trn_inds = np.where(labels == cl_ind)[0]
    tst_inds = np.where(labels_t == cl_ind)[0]
    ho_inds = np.where(labels_ho == cl_ind)[0]
    pois_inds = np.random.choice(
        ho_inds,
        int(ho_inds.shape[0] * pois_rate),
        replace=True
    )
    print("cluster ind:", cl_ind)
    print("train cluster size:", trn_inds.shape[0])
    print("test cluster size:", tst_inds.shape[0])
    print("pois cluster size", pois_inds.shape[0])
    trn_x = x
    trn_y = y
    trn_x_att = x_att

    preds_cl = preds_ho[ho_inds].sum(axis=0)
    assert preds_cl.size == 2

    worst_class = np.argmin(preds_cl)
    print(worst_class, preds_cl)

    pois_x = np.take(x_ho, pois_inds, axis=0)
    pois_y = np.take(y_ho, pois_inds, axis=0)
    pois_x_att = np.take(x_ho_att, pois_inds, axis=0)

    print(pois_y)
    pois_y[:] = worst_class  # Assigns the worst class label to every poison point
    print(pois_y)

    trn_x = np.concatenate((trn_x, pois_x))
    trn_y = np.concatenate((trn_y, pois_y))
    trn_x_att = np.concatenate((trn_x_att, pois_x_att))
    xt_p, xt_p_att, yt_p = x_t[tst_inds], x_t_att[tst_inds], y_t[tst_inds]

    # Create the subset of the test set not containing the targeted
    # sub population to compute the collateral damage
    x_coll = x_t[[i for i in range(x_t.shape[0]) if i not in tst_inds]]
    x_coll_att = x_t_att[[i for i in range(x_t_att.shape[0]) if i not in tst_inds]]
    y_coll = y_t[[i for i in range(y_t.shape[0]) if i not in tst_inds]]

    pth = os.path.join(common.storage_dir, 'imdb_bert_{}_pop_{}'.format('LL' if frozen else 'FT', cl_ind))
    if not os.path.isdir(pth):
        os.makedirs(pth)

    np.save(os.path.join(pth, 'pois_x_{}.npy'.format(cl_ind)), pois_x)
    np.save(os.path.join(pth, 'pois_x_att_{}.npy'.format(cl_ind)), pois_x_att)
    np.save(os.path.join(pth, 'pois_y_{}.npy'.format(cl_ind)), pois_y)

    np.save(os.path.join(pth, 'trn_x_{}.npy'.format(cl_ind)), trn_x)
    np.save(os.path.join(pth, 'trn_x_att_{}.npy'.format(cl_ind)), trn_x_att)
    np.save(os.path.join(pth, 'trn_y_{}.npy'.format(cl_ind)), trn_y)

    np.save(os.path.join(pth, 'x_t_{}.npy'.format(cl_ind)), x_t)
    np.save(os.path.join(pth, 'x_t_att_{}.npy'.format(cl_ind)), x_t_att)
    np.save(os.path.join(pth, 'y_t_{}.npy'.format(cl_ind)), y_t)

    np.save(os.path.join(pth, 'xt_p_{}.npy'.format(cl_ind)), xt_p)
    np.save(os.path.join(pth, 'xt_p_att_{}.npy'.format(cl_ind)), xt_p_att)
    np.save(os.path.join(pth, 'yt_p_{}.npy'.format(cl_ind)), yt_p)

    np.save(os.path.join(pth, 'x_coll_{}.npy'.format(cl_ind)), x_coll)
    np.save(os.path.join(pth, 'x_coll_att_{}.npy'.format(cl_ind)), x_coll_att)
    np.save(os.path.join(pth, 'y_coll_{}.npy'.format(cl_ind)), y_coll)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_clusters", help="number of clusters", type=int, default=100)
    parser.add_argument("--pca_dim", help="projection dimension for PCA", type=int, default=10)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument('--poison_rate', help='poisoning rate', type=float, default=0.5)
    parser.add_argument('--frozen', action='store_true', help='fine tunes only BERT last layer and classifier')
    parser.add_argument("--class_ind", help="target class index", type=int, required=True)

    arguments = vars(parser.parse_args())
    attack(arguments)
