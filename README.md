# Subpopulation Data Poisoning Attacks

Code for "Subpopulation Data Poisoning Attacks"

Code for experiments with text data can be found in ...

Code for image experiments can be found in image/ 


UCI Adult experiments can be found at this colab notebook: https://colab.research.google.com/drive/1qWZeEnzxO9P9lLpjIsvBSdTBZwNDK02I



## Experiments with sentiment classification

To reproduce the results on the sentiment classification IMDB data run the 
following scripts.

Note: in all the scripts it is important to leave the seed to the default value. If changed,
please make sure to use the same value for the seed in all scripts. It is used to split the 
training data in the defender's and attacker's sets. 

#### Training
First fine tune the bert model used by the attacker: 

```shell script
python train_imdb_bert.py
```

Here the option `--frozen` can be used to specify fine tuning the "small" or Last Layer (LL) model. This will only tune 
the last attention layer and the classification layer.

Note: this script will also pre-process and tokenize the data if it doesn't find the DataFrames in the file system.

##### Evaluation

The notebook `notebooks/evaluations/bert_base_evaluation.ipynb` can be used to evaluate the fine tuned models on 
the IMDB test data.

#### Representations
Extract the representations of the defender's training set, adversary's training, and test set used in the attack 
by running:
 
```shell script
python representations_imdb_bert.py
```

Again the `--frozen` flag can be used to work with the LL model. 

This script assumes the trained model and data files are already available.
 

#### Attack
To run the sub-population poisoning attack on BERT over the IMDB data run the script `attack_nlp.py`:

```shell script
usage: attack_nlp.py [-h] [--batch BATCH] [--epochs EPOCHS]
                     [--n_clusters N_CLUSTERS] [--pca_dim PCA_DIM]
                     [--seed SEED] [--learning_rate LEARNING_RATE]
                     [--poison_rate POISON_RATE] [--frozen] [--all]

optional arguments:
  -h, --help            show this help message and exit
  --batch BATCH         number of samples per batch
  --epochs EPOCHS       number of epochs of fine tuning
  --n_clusters N_CLUSTERS
                        number of clusters
  --pca_dim PCA_DIM     projection dimension for PCA
  --seed SEED           random seed
  --learning_rate LEARNING_RATE
                        learning rate
  --poison_rate POISON_RATE
                        poisoning rate
  --frozen              fine tunes only BERT last layer and classifier
  --all                 fine tunes BERT using the entire dataset
```

For instance to run the attack on BERT-FT with 100 clusters and a poisoning rate of 0.5:
```shell script
python attack_nlp.py --poison_rate 0.5 --n_clusters 100
``` 

For convenience, the shell script `run_all_bert.sh` can be used to run all BERT-IMDB experiments present in the paper.
Experimental results will be accumulated in `results/bert`.


#### Results

The statistics on the effect of subpopulation attacks on BERT models can be explored using the notebook: 
`notebooks/evaluations/explore_bert_results.ipynb`.

#### TRIM mitigation

Experiments on using TRIM to mitigate subpopulation attacks on BERT models over the IMDB dataset
can be replicated using the notebooks:
`notebooks/evaluations/trim_defense_bert_ft.ipynb`, 
`notebooks/evaluations/trim_defense_bert_ll.ipynb`.