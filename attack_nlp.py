import os
import gc
import argparse

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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
    batch = args['batch']
    epochs = args['epochs']
    n_clusters = args['n_clusters']
    pca_dim = args['pca_dim']
    seed = args['seed']
    lr = args['learning_rate']
    frozen = args['frozen']
    pois_rate = args['poison_rate']
    n_eval = args['n_eval']

    # Initialization
    device = bert_utils.get_device()  # Check if cuda available
    bert_utils.set_seed(device, seed=seed)  # Seed all the PRNGs
    model_name_def = 'imdb_bert_{}_DEF'.format('LL' if frozen else 'FT')

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

    cl_conf = np.zeros((n_clusters,))
    for cl_ind in range(n_clusters):
        cl_inds = np.where(labels_ho == cl_ind)[0]
        cl_preds = preds_ho[cl_inds]
        cl_conf[cl_ind] = np.multiply(cl_preds, np.eye(2)[y_ho[cl_inds]]).mean()

    conf_ordered = np.argsort(cl_conf)
    all_inds = conf_ordered[:n_eval].ravel().tolist() + \
        conf_ordered[n_clusters // 2 - (n_eval // 2):n_clusters // 2 + (n_eval // 2)].ravel().tolist() + \
        conf_ordered[-n_eval:].ravel().tolist()
    print('Indices of clusters to evaluate: {}\n{}\n'.format(len(all_inds), all_inds))

    all_eval_stats = {}

    for cl_ind in all_inds:
        gc.collect()
        torch.cuda.empty_cache()

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

        pois_y[:] = worst_class  # Assigns the worst class label to every poison point
        trn_x = np.concatenate((trn_x, pois_x))
        trn_y = np.concatenate((trn_y, pois_y))
        trn_x_att = np.concatenate((trn_x_att, pois_x_att))
        rand_inds = np.random.choice(trn_x.shape[0], trn_x.shape[0], replace=False)
        xt_p, xt_p_att, yt_p = x_t[tst_inds], x_t_att[tst_inds], y_t[tst_inds]

        # Create the subset of the test set not containing the targeted
        # sub population to compute the collateral damage
        x_coll = x_t[[i for i in range(x_t.shape[0]) if i not in tst_inds]]
        x_coll_att = x_t_att[[i for i in range(x_t_att.shape[0]) if i not in tst_inds]]
        y_coll = y_t[[i for i in range(y_t.shape[0]) if i not in tst_inds]]
        print('\nx coll shape: {}\nx_att coll shape:{}\ny coll shape: {}'.format(
            x_coll.shape, x_coll_att.shape, y_coll.shape))

        print('Training new model')
        save_path = os.path.join(
            common.saved_models_dir,
            'victim_bert_{}'.format(cl_ind)
        )
        model = bert_utils.wrap_train(
            trn_x[rand_inds],
            trn_y[rand_inds],
            trn_x_att[rand_inds],
            b_size=batch,
            lr=lr,
            epochs=epochs,
            frozen=frozen,
            save=save_path
        )
        stats = bert_utils.eval_stats(
            model=model,
            trn_x=trn_x[rand_inds],
            trn_x_att=trn_x_att[rand_inds],
            trn_y=trn_y[rand_inds],
            x_t=x_t,
            x_t_att=x_t_att,
            y_t=y_t,
            xt_p=xt_p,
            xt_p_att=xt_p_att,
            yt_p=yt_p,
            b_size=batch
        )
        all_eval_stats[cl_ind] = stats

        # Collateral accuracy evaluation on the attacked model
        pois_coll_acc = bert_utils.get_accuracy_bert(model=model, x=x_coll, x_att=x_coll_att, y=y_coll, b_size=batch)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Save information about cluster size
        all_eval_stats[cl_ind]['train_clus_size'] = trn_inds.shape
        all_eval_stats[cl_ind]['test_clus_size'] = tst_inds.shape
        all_eval_stats[cl_ind]['pois_clus_size'] = pois_inds.shape

        # Compute base accuracy for defender model on the test data
        model_def = bert_utils.load_bert(model_file=model_name_def + '.ckpt')
        pois_ds = TensorDataset(
            torch.from_numpy(xt_p),
            torch.from_numpy(xt_p_att),
            torch.from_numpy(yt_p)
        )
        pois_dl = DataLoader(pois_ds, shuffle=False, batch_size=batch)
        accuracy_def = bert_utils.predict_bert(model_def, device, pois_dl, acc=True)
        all_eval_stats[cl_ind]['base_def'] = accuracy_def

        # Collateral accuracy evaluation on the defender model, and collateral damage estimation
        def_coll_acc = bert_utils.get_accuracy_bert(model=model_def, x=x_coll, x_att=x_coll_att, y=y_coll, b_size=batch)
        coll_dmg = def_coll_acc - pois_coll_acc
        all_eval_stats[cl_ind]['collateral_dmg'] = coll_dmg

        print('Eval stats: {}\n\n'.format(all_eval_stats[cl_ind]))
        del model_def
        gc.collect()
        torch.cuda.empty_cache()

    res_path = common.results_dir_bert
    res_file = 'eval-stats_clus{}_pois{}_{}'.format(n_clusters, pois_rate, 'LL' if frozen else 'FT')
    np.save(os.path.join(res_path, res_file), all_eval_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', help='number of samples per batch', type=int, default=8)
    parser.add_argument('--epochs', help='number of epochs of fine tuning', type=int, default=4)
    parser.add_argument("--n_clusters", help="number of clusters", type=int, default=100)
    parser.add_argument("--n_eval", help="number of clusters to evaluate - must be even", type=int, default=10)
    parser.add_argument("--pca_dim", help="projection dimension for PCA", type=int, default=10)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=1e-5)
    parser.add_argument('--poison_rate', help='poisoning rate', type=float, default=0.5)
    parser.add_argument('--frozen', action='store_true', help='fine tunes only BERT last layer and classifier')
    parser.add_argument('--all', action='store_true', help='fine tunes BERT using the entire dataset')

    arguments = vars(parser.parse_args())
    attack(arguments)
