import argparse
import os.path

import numpy as np
import time

from environments.roulette import RouletteEnv
from environments.gridworld import GridworldEnv
from agents.q import Qlearning
from agents.doubleq import DQlearning
from agents.weighted import WDQlearning
from agents.maxmin import MaxMinQlearning
from agents.randmin import RandMinQlearning
from agents.avgmin import AvgMinQlearning


def get_agent(env, args):
    if args.agent == "q":
        agent = Qlearning(env, args.gamma, args.lr, args.eps)
    elif args.agent == "doubleq":
        agent = DQlearning(env, args.gamma, args.lr, args.eps)
    elif args.agent == "wdq":
        agent = WDQlearning(env, args.wdq_c, args.gamma, args.lr, args.eps)
    elif args.agent == "maxmin":
        agent = MaxMinQlearning(env, args.maxmin_n, args.gamma, args.lr, args.eps)
    elif args.agent == "randmin":
        agent = RandMinQlearning(env, args.randmin_p, args.gamma, args.lr, args.eps)
    elif args.agent == "avgmin":
        agent = AvgMinQlearning(env, args.avgmin_beta, args.gamma, args.lr, args.eps)
    return agent


def set_seed(env, seed):
    env.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        "--agent",
        type=str,
        default="q",
        choices=["q", "doubleq", "wdq", "maxmin", "randmin", "avgmin"],
    )
    parser.add_argument(
        "--env",
        "--env",
        type=str,
        default="roulette",
        choices=["roulette", "gridworld"],
    )
    parser.add_argument("--env_var1", "--env_var1", type=int, default=0)
    parser.add_argument("--env_var2", "--env_var2", type=int, default=0)
    parser.add_argument("--agent_nb", "--agent_nb", type=int, default=2)
    parser.add_argument("--wdq_c", "--wdq_c", type=int, default=1, choices=[1, 10, 100])
    parser.add_argument(
        "--maxmin_n", "--maxmin_n", type=int, default=2, choices=[2, 4, 6, 8]
    )
    parser.add_argument("--randmin_p", "--randmin_p", type=float, default=0.1)
    parser.add_argument("--avgmin_beta", "--avgmin_beta", type=float, default=0.1)
    parser.add_argument("--lr", "--lr", type=float, default=0.08)
    parser.add_argument("--n_ep", "--n_episodes", type=int, default=150)
    parser.add_argument("--maxt", "--env_max_steps", type=int, default=500)
    parser.add_argument("--itv", "--interval", type=int, default=50)
    parser.add_argument("--eps", "--eps", type=float, default=0.1)
    parser.add_argument("--gamma", "--gamma", type=float, default=0.95)
    parser.add_argument("--thres", "--threshold", type=float, default=0.01)
    parser.add_argument("--n_runs", "--n_runs", type=int, default=100)
    parser.add_argument("--verbose", "--v", action="store_true")
    parser.add_argument("--render", "--render", action="store_true")
    args = parser.parse_args()

    if args.env == "roulette":
        env = RouletteEnv(max_steps=args.maxt)
    else:
        env = GridworldEnv(
            n=3, max_steps=args.maxt, var1=args.env_var1, var2=args.env_var2
        )

    steps = np.empty((args.n_runs, args.n_ep))
    rewards = np.empty((args.n_runs, args.n_ep))
    diffs = []
    times = []
    for i_run in range(args.n_runs):
        set_seed(env, i_run)
        t_start = time.time()
        agent = get_agent(env, args)
        cur_reward, cur_step, diff = agent.train(
            args.n_ep, args.itv, args.thres, args.render, args.verbose
        )
        rewards[i_run] = np.array(cur_reward)
        steps[i_run] = np.array(cur_step)
        diffs.append(diff)
        times.append(time.time() - t_start)

    np.save(os.path.join("result", env.name, agent.name + "_step.npy"), steps)
    np.save(os.path.join("result", env.name, agent.name + "_reward.npy"), rewards)

    f = open(os.path.join("result", env.name + ".txt"), "a")
    f.write(str(args) + "\n")
    f.write(
        "{}:\t diff {:.2f} $\\pm$ {:.2f} \t reward {:.2f} $\\pm$ {:.2f} \t time {:.2f} $\\pm$ {:.2f}".format(
            agent.name,
            np.mean(diff),
            np.std(diff),
            np.mean(rewards[:, -1]),
            np.std(rewards[:, -1]),
            np.mean(times),
            np.std(times),
        )
    )
    f.close()


if __name__ == "__main__":
    main()
