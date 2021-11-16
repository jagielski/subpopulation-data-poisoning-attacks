#!/bin/bash

# time python attack_nlp.py --frozen --poison_rate 0.5 --n_eval 10;
# time sleep 1;
# time python attack_nlp.py --frozen --poison_rate 1.0 --n_eval 10;
# time sleep 1;
# time python attack_nlp.py --frozen --poison_rate 2.0 --n_eval 10;
# time sleep 1;
# time python attack_nlp.py --poison_rate 0.5 --n_eval 10;
# time sleep 1;
# time python attack_nlp.py --poison_rate 1.0 --n_eval 10;
# time sleep 1;
time python attack_nlp.py --poison_rate 2.0 --n_eval 10;
