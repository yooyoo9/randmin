#!/bin/bash

echo 'Running experiments for Gridworld(1,1)...'
python main.py --env=gridworld --env_var1=1 --env_var2=1 --agent=q --lr=0.02
python main.py --env=gridworld --env_var1=1 --env_var2=1 --agent=doubleq --lr=0.01
python main.py --env=gridworld --env_var1=1 --env_var2=1 --agent=wdq --wdq_c=10 --lr=0.01
python main.py --env=gridworld --env_var1=1 --env_var2=1 --agent=maxmin --maxmin=4 --lr=0.04
python main.py --env=gridworld --env_var1=1 --env_var2=1 --agent=randmin --randmin_p=0.1 --lr=0.08
python main.py --env=gridworld --env_var1=1 --env_var2=1 --agent=avgmin --avgmin_beta=0.3 --lr=0.04

echo 'Running experiments for Gridworld(1,10)...'
python main.py --env=gridworld --env_var1=1 --env_var2=10 --agent=q --lr=0.02
python main.py --env=gridworld --env_var1=1 --env_var2=10 --agent=doubleq --lr=0.08
python main.py --env=gridworld --env_var1=1 --env_var2=10 --agent=wdq --wdq_c=100 --lr=0.02
python main.py --env=gridworld --env_var1=1 --env_var2=10 --agent=maxmin --maxmin=6 --lr=0.01
python main.py --env=gridworld --env_var1=1 --env_var2=10 --agent=randmin --randmin_p=0.2 --lr=0.08
python main.py --env=gridworld --env_var1=1 --env_var2=10 --agent=avgmin --avgmin_beta=0.3 --lr=0.08

echo 'Running experiments for Gridworld(10,1)...'
python main.py --env=gridworld --env_var1=10 --env_var2=1 --agent=q --lr=0.01
python main.py --env=gridworld --env_var1=10 --env_var2=1 --agent=doubleq --lr=0.08
python main.py --env=gridworld --env_var1=10 --env_var2=1 --agent=wdq --wdq_c=100 --lr=0.04
python main.py --env=gridworld --env_var1=10 --env_var2=1 --agent=maxmin --maxmin=2 --lr=0.02
python main.py --env=gridworld --env_var1=10 --env_var2=1 --agent=randmin --randmin_p=0.5 --lr=0.01
python main.py --env=gridworld --env_var1=10 --env_var2=1 --agent=avgmin --avgmin_beta=0.2 --lr=0.01

echo 'Running experiments for Gridworld(10,10)...'
python main.py --env=gridworld --env_var1=10 --env_var2=10 --agent=q --lr=0.08
python main.py --env=gridworld --env_var1=10 --env_var2=10 --agent=doubleq --lr=0.02
python main.py --env=gridworld --env_var1=10 --env_var2=10 --agent=wdq --wdq_c=100 --lr=0.04
python main.py --env=gridworld --env_var1=10 --env_var2=10 --agent=maxmin --maxmin=8 --lr=0.02
python main.py --env=gridworld --env_var1=10 --env_var2=10 --agent=randmin --randmin_p=0.3 --lr=0.08
python main.py --env=gridworld --env_var1=10 --env_var2=10 --agent=avgmin --avgmin_beta=0.5 --lr=0.08
