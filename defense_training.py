import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.cluster import KMeans

from sys import argv

import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--dataset', choices=['utk', 'cifar'])
parser.add_argument('--size', choices=['small', 'large'])
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--layer', type=int)
parser.add_argument('--pca', action='store_true')
parser.add_argument('--km', action='store_true')
parser.add_argument('--fm', action='store_true')
parser.add_argument('--exp_ind', type=int)
parser.add_argument('--defense', choices=['trim', 'sever'])
args = parser.parse_args()
print(args)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

def model_fn(dataset, size):
    tf.compat.v1.reset_default_graph()
    if dataset=='cifar':
        shape = (32, 32, 3)
        n_classes = 10
        if size=='small':
            model = tf.keras.models.Sequential()
            scales = 3
            reg = tf.keras.regularizers.l2(l=0.00)
            model.add(tf.keras.layers.InputLayer(shape))
            model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                kernel_regularizer=reg))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
            for scale in range(scales):
                model.add(tf.keras.layers.Conv2D(32 << scale, (3, 3), padding='same',
                    kernel_regularizer=reg))
                model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
                model.add(tf.keras.layers.Conv2D(64 << scale, (3, 3), padding='same',
                    kernel_regularizer=reg))
                model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
                model.add(tf.keras.layers.AveragePooling2D((2, 2)))
            model.add(tf.keras.layers.Conv2D(n_classes, (3, 3), padding='same',
                    kernel_regularizer=reg))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

            #model.add(tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=[1, 2])))
            #model.add(tf.keras.layers.Softmax())
            
            opt = tf.keras.optimizers.Adam(lr=0.001)  # SGD(0.002, momentum=.5)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            return model
    else:
        shape = (100, 100, 3)
        n_classes = 2
    vgg = tf.keras.applications.VGG16(include_top=False, input_shape=shape, pooling='avg')
    if size=='small':
        opt = tf.keras.optimizers.Adam(0.001)
        for layer in vgg.layers:
            layer.trainable = False
    else:
        opt = tf.keras.optimizers.Adam(0.0001)  # SGD(0.01, momentum=.9)

    output = tf.keras.layers.Dense(n_classes, kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
            activation='softmax')(vgg.output)
    model = tf.keras.models.Model(inputs=vgg.inputs[0], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def trim(dataset, size, x, y, num_remove):
    inds = []
    new_inds = list(range(x.shape[0]))
    it = 0
    while sorted(new_inds) != sorted(inds) and it < 5:
        it += 1
        tf.keras.backend.clear_session()
        inds = new_inds[:]
        model = train_model(dataset, size, x[inds], y[inds], x[inds], y[inds])
        preds = model.predict(x)
        probs = np.multiply(preds, y).sum(axis=1)
        new_inds = np.argpartition(probs, num_remove)[num_remove:]
        print(probs[np.argpartition(probs, num_remove)[num_remove:]].mean(), probs[np.argpartition(probs, num_remove)[:num_remove:]].mean())
    return model, new_inds

def sever(dataset, size, x, y, fea, num_remove):
    pca = PCA(1)
    inds = []
    new_inds = list(range(x.shape[0]))
    it = 0
    while sorted(new_inds) != sorted(inds) and it < 5:
        it += 1
        tf.keras.backend.clear_session()
        inds = new_inds[:]
        model = train_model(dataset, size, x[inds], y[inds], x[inds], y[inds])
        preds = model.predict(x)
        probs = np.multiply(preds, y).sum(axis=1)
        losses = np.multiply(probs, 1-probs)
        norms = np.linalg.norm(fea, axis=1)
        assert norms.shape[0]==x.shape[0]
        scores = np.square(losses * norms)
        new_inds = np.argpartition(scores, -num_remove)[:-num_remove]
        print(scores[new_inds].mean(), scores.mean())
    return model, new_inds

def train_model(dataset, size, x, y, tst_x, tst_y):
    model = model_fn(dataset, size)
    if dataset=='cifar':
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
        datagen.fit(x)
        model.fit_generator(datagen.flow(x, y, batch_size=32),
                        epochs=12, validation_data=(tst_x, tst_y))
    else:
        assert dataset=='utk'
        model.fit(x, y, epochs=12, batch_size=32, validation_data=(tst_x, tst_y))
    return model

if args.dataset=='utk':
    PCA_DIM = 2000
    C = 0.0001
    img_pp = np.load('data/utk_imgs.npy')
    fea_pp = np.load('data/utk_preds.npy')
    cl_pp = np.load('data/utk_classes.npy')
    include_inds = np.where(cl_pp[:, 0] >= 15)[0]
    fea = fea_pp[include_inds]
    cl = cl_pp[include_inds]
    img = img_pp[include_inds]
    print(img.shape)
    target = np.eye(2)[cl[:, 1]]  # 0 == age, 1 == gender, 2 == race
    age_buckets = [30, 45, 60]
    ages = np.array([cl[:, 0] >= b for b in age_buckets]).sum(axis=0)
    races = cl[:, 2]
    target_name = races*4 + ages
    print(np.unique(target_name, return_counts=True))
    trn_size = 7000
    aux_size = 7000
    tst_size = fea.shape[0] - trn_size - aux_size
    np.random.seed(0)
    inds_shuffle = np.random.choice(fea.shape[0], fea.shape[0])
    trn_inds = inds_shuffle[:trn_size]
    aux_inds = inds_shuffle[trn_size:trn_size + aux_size]
    tst_inds = inds_shuffle[trn_size + aux_size:]

    fea_trn, targ_trn, tn_trn = fea[trn_inds], target[trn_inds], target_name[trn_inds]
    fea_aux, targ_aux, tn_aux = fea[aux_inds], target[aux_inds], target_name[aux_inds]
    fea_tst, targ_tst, tn_tst = fea[tst_inds], target[tst_inds], target_name[tst_inds]

    img_trn, img_aux, img_tst = img[trn_inds], img[aux_inds], img[tst_inds]
    print(img_trn.shape, fea_trn.shape, targ_trn.shape)

elif args.dataset=='cifar':
    (x_trn, y_trn), (img_tst, targ_tst) = tf.keras.datasets.cifar10.load_data()
    print(x_trn.shape, y_trn.shape, img_tst.shape, targ_tst.shape, x_trn.max(), x_trn.min())
    x_trn = x_trn.astype(np.float)/255 - 0.5
    img_tst = img_tst.astype(np.float)/255 - 0.5
    y_trn = np.eye(10)[y_trn.ravel()]
    targ_tst = np.eye(10)[targ_tst.ravel()]
    np.random.seed(0)
    inds = np.random.choice(x_trn.shape[0], x_trn.shape[0], replace=False)
    trn_inds = inds[:x_trn.shape[0]//2]
    aux_inds = inds[x_trn.shape[0]//2:]
    img_trn, targ_trn = x_trn[trn_inds], y_trn[trn_inds]
    img_aux, targ_aux = x_trn[aux_inds], y_trn[aux_inds]


def batch_fn(fn, inp, b):
    all_out = []
    start_ind = range(0, inp.shape[0], b)
    for inp_ind in start_ind:
        all_out.append(fn(inp[inp_ind:inp_ind+b]))
    return np.concatenate(all_out)


def gradient_match(model_path, pert_x, pert_y, targ_x, targ_y):
    model = tf.keras.models.load_model(model_path)
    orig_pert_x, orig_pert_y = pert_x.copy(), pert_y.copy()
    tf_x, tf_y = tf.Variable(pert_x), tf.Variable(pert_y)
    tf_targx, tf_targy = tf.Variable(targ_x), tf.Variable(targ_y)
    loss_obj = tf.keras.losses.categorical_crossentropy
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        targ_loss = tf.reduce_mean(loss_obj(tf_targy, model(tf_targx)))
    targ_grads = tape.gradient(targ_loss, model.trainable_weights[-2:])
    print("TARG")
    print(targ_loss)
    print(targ_grads) 
    lr = .2
    its = 50
    for _ in range(its):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf_x)
            loss = tf.reduce_mean(loss_obj(model(tf_x), tf_y))

            grads = tape.gradient(loss, model.trainable_weights[-2:])
            opt_loss = tf.reduce_sum([tf.reduce_sum(tf.math.multiply(g, tg)) for (g, tg) in zip(grads, targ_grads)])
            print(opt_loss)

        x_grad = tape.gradient(opt_loss, tf_x)
        new_val = tf_x.value() - x_grad*lr
        pert_x = np.clip(new_val, orig_pert_x - 1/510, orig_pert_x + 1/510)
        tf.keras.backend.set_value(tf_x, pert_x)
    print(np.linalg.norm((pert_x - orig_pert_x).reshape(pert_x.shape[0], -1), axis=1).mean())
    
    return pert_x

pretrained_path = f"/net/data/pois_priv/subclass_avail_cifar/{args.dataset}_pretrained_{args.size}"

if not args.pretrained:
    model = train_model(args.dataset, args.size, img_aux, targ_aux, img_tst, targ_tst)
    model.save(pretrained_path)

with tf.compat.v1.Session() as sess:
    model = tf.keras.models.load_model(pretrained_path)
    all_layers = [model.inputs[0]]
    for layer in model.layers:
        if args.dataset=='cifar' and args.size=='small':
            if layer.name.startswith('average_pooling') or layer.name.startswith('flatten'):
                all_layers.append(layer)
        else:
            if 'pool' in layer.name:
                all_layers.append(layer)

    layer = (all_layers[args.layer].output if args.layer != 0 else model.inputs[0])
    layer_fn = lambda inp: sess.run(layer, feed_dict={model.inputs[0]: inp})
    fea_trn = batch_fn(layer_fn, img_trn, 100)
    fea_aux = batch_fn(layer_fn, img_aux, 100)
    fea_tst = batch_fn(layer_fn, img_tst, 100)

prefix = '_'.join([args.dataset, args.size, str(args.layer), ('fm' if args.fm else 'cm')])

PCA_DIM, KM_NUM, KM_DIM = 100, 100, 40
if args.pca:
  pca = PCA(PCA_DIM)
  pca.fit(fea_aux.reshape((fea_aux.shape[0], -1)))
  pickle.dump(pca, open(prefix+"_pca.p", 'wb'))
  pca_trn, pca_aux, pca_tst = pca.transform(fea_trn.reshape((fea_trn.shape[0], -1))), pca.transform(fea_aux.reshape((fea_aux.shape[0], -1))), pca.transform(fea_tst.reshape((fea_tst.shape[0], -1)))
  np.save(prefix+"_pca_"+str(args.layer), [pca_trn, pca_aux, pca_tst])
  print("trained pca")
else:
  pca = pickle.load(open(prefix+"_pca.p", 'rb'))
  pca_trn, pca_aux, pca_tst = np.load(prefix+"_pca_" + str(args.layer) + ".npy", allow_pickle=True)
  print("loaded pca")

print("pca_var", PCA_DIM, pca.explained_variance_ratio_.sum())


if args.km:
  km = KMeans(KM_NUM)
  km.fit(pca_aux[:, :KM_DIM])
  pickle.dump(km, open(prefix+"_km_" + str(args.layer) + ".p", 'wb'))
  print("trained km")
else:
  km = pickle.load(open(prefix+"_km_" + str(args.layer) + ".p", 'rb'))
  print("loaded km")

if args.fm:
  print("using fm")
  trn_cl, aux_cl, tst_cl = tn_trn, tn_aux, tn_tst
else:
  print("using cm")
  trn_cl = km.predict(pca_trn[:, :KM_DIM])
  aux_cl = km.predict(pca_aux[:, :KM_DIM])
  tst_cl = km.predict(pca_tst[:, :KM_DIM])

all_cl = np.unique(aux_cl)

if args.exp_ind==0:
    accs = np.zeros_like(all_cl, dtype=np.float32)
    for _ in range(5):
        model = train_model(args.dataset, args.size, img_trn, targ_trn, img_tst, targ_tst)
        for cl_ind in all_cl:
            tst_cl_inds = np.where(tst_cl==cl_ind)[0]
            if tst_cl_inds.size==0:
                print("too small")
                accs[cl_ind] = -1
                continue
            cl_tst_preds = model.predict(img_tst[tst_cl_inds])
            cl_tst_targs = targ_tst[tst_cl_inds]
            print(cl_tst_preds.shape)
            print(cl_tst_targs.shape)
            acc = (cl_tst_preds.argmax(1)==cl_tst_targs.argmax(1)).mean()
            accs[cl_ind] += acc

    np.save(prefix+'_accs_'+str(args.layer), accs/5) 


    confs = np.zeros_like(all_cl, dtype=np.float32)
    sizes = np.zeros_like(all_cl, dtype=np.float32)
    for _ in range(5):
        model = train_model(args.dataset, args.size, img_aux, targ_aux, img_tst, targ_tst)
        for cl_ind in all_cl:
            aux_cl_inds = np.where(aux_cl==cl_ind)[0]
            sizes[cl_ind] = aux_cl_inds.size
        
            cl_aux_preds = model.predict(img_aux[aux_cl_inds])
            cl_aux_targs = targ_aux[aux_cl_inds]
        
            conf = np.multiply(cl_aux_preds, cl_aux_targs).mean()*2
            confs[cl_ind] += conf
        
            print(cl_ind, cl_tst_preds.shape, conf)

    np.save(prefix+'_confs_'+str(args.layer), confs/5) 
    np.save(prefix+'_sizes_'+str(args.layer), sizes) 
else:
    confs = np.load(prefix+"_confs_" + str(args.layer) + ".npy")

low_confs = np.argsort(confs)
print(confs)


num_to_att = KM_NUM
cl_to_att = low_confs[:num_to_att]

pois_rates = [0.5, 1, 2]  # pois_rates = [0.5, 1, 2]  # [0.5, 1, 2]

from itertools import product

all_exp = sorted(list(product(range(3), pois_rates)))
trn_ind, pois_rate = all_exp[args.exp_ind]
for fake_index in range(1):
    #for trn_ind, pois_rate in product(range(5), pois_rates):
    errs = np.zeros((all_cl.shape[0],), dtype=np.float32)
    collats = np.zeros((all_cl.shape[0],), dtype=np.float32)
    pois_rate_ind = pois_rates.index(pois_rate)
    for cl_ind in cl_to_att:
        tf.keras.backend.clear_session()
        trn_inds = np.where(trn_cl==cl_ind)[0]
        cl_inds = np.where(aux_cl==cl_ind)[0]
        tst_inds = np.where(tst_cl==cl_ind)[0]
        if tst_inds.size==0:
            print("clind {} too small".format(tst_inds.size), trn_inds.size)
            continue
        aux_img = img_aux[cl_inds]
        aux_y = targ_aux[cl_inds]

        worst_class = np.argmin(aux_y.mean(axis=0))
    
        pois_inds = np.random.choice(aux_img.shape[0], int(pois_rate*aux_img.shape[0]), replace=True)

        p_x = aux_img[pois_inds]
        p_y = np.zeros_like(aux_y)[pois_inds]
        p_y[:, worst_class] = 1

        #px_gm = gradient_match(pretrained_path, p_x, p_y, img_trn[trn_inds], targ_trn[trn_inds])

        p_trn_x, p_trn_y = np.concatenate((img_trn, p_x)), np.concatenate((targ_trn, p_y))
        #p_trn_x, p_trn_y = np.concatenate((img_trn, px_gm)), np.concatenate((targ_trn, p_y))

        all_fea = np.concatenate((fea_trn, fea_aux[cl_inds][pois_inds])) 

        if args.defense=='trim':
            model, inds = trim(args.dataset, args.size, p_trn_x, p_trn_y, p_x.shape[0])  # num_remove)
        elif args.defense=='sever':
            model, inds = sever(args.dataset, args.size, p_trn_x, p_trn_y, all_fea, p_x.shape[0])  # num_remove)

        # train_model(args.dataset, args.size, p_trn_x, p_trn_y, img_tst, targ_tst)

        clean_acc = (model.predict(img_tst).argmax(1)==targ_tst.argmax(1)).mean()
        pois_acc = (model.predict(img_tst[tst_inds]).argmax(1)==targ_tst[tst_inds].argmax(1)).mean()
        errs[cl_ind] = pois_acc
        collats[cl_ind] = clean_acc
        print("pois {} rate {}: trn {}, tst {}, clean acc {:.3f}, pois acc {:.3f}".format(cl_ind, pois_rate, trn_inds.shape[0], tst_inds.shape[0], clean_acc, pois_acc))

        np.save(prefix + "_" + str(args.layer) + "_" + args.defense + "_errs_" + str(trn_ind)+'_'+str(pois_rate), errs)
        np.save(prefix + "_" + str(args.layer) + "_" + args.defense + "_cleans_" + str(trn_ind)+'_'+str(pois_rate), collats)
