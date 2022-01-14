#!/bin/bash
echo 'Running experiments for Roulette...'
python main.py --env=roulette --agent=q
python main.py --env=roulette --agent=doubleq --lr=0.01
python main.py --env=roulette --agent=wdq --wdq_c=1 --lr=0.02
python main.py --env=roulette --agent=maxmin --maxmin_n=2 --lr=0.08
python main.py --env=roulette --agent=randmin --randmin_p=0.1 --lr=0.08
python main.py --env=roulette --agent=avgmin --avgmin_beta=0.1 --lr=0.08
