# RandMin Q-learning: Controlling Overestimation Bias
2021 Reinforcement Learning Course, Project at ETH Zurich, [paper link](https://github.com/yooyoo9/randmin/blob/main/paper.pdf)

### Abstract:
Q-learning is known to suffer from overestimation bias as it approximates the maximum action value using the maximum estimated action value.
Several algorithms have been proposed to control overestimation bias but they either introduce underestimation bias or make use of multiple action value estimates which is computationally not efficient.
In this paper, we propose one generalization of Q-learning called \textit{RandMin Q-learning} which maintains two estimates of the action value function and provides a parameter to control the estimation bias.
We show the convergence of our algorithm in the tabular case and generalize its idea to the function approximation setting: \textit{RandMin DQN}.
We empirically verify that our algorithm achieves superior performance on several highly stochastic problems in the tabular case but also in environments with a continuous state space.

### How to Run Experiments:
Set up environment for the first time:
```
pip install -r requirements.txt
```

To run experiments, go to the root directory and type (the default parameter can be used for result reproduction):
```
python main.py
```
The results are stored in the folder **./results**.
