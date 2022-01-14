import os
import random
import torch
import numpy as np
import argparse

from agents.dqn import DQN
from agents.ddqn import DDQN
from agents.maxmindqn import MaxminDQN
from agents.randmin import RandMin
from agents.averaged import AveragedDQN
from environments.mountain_car import RandMountainCarEnv
from environments.cartpole import RandCartPoleEnv
from environments.acrobot import RandAcrobotEnv

N_EPISODES = 400  # 1000
N_RUNS = 5  # 20
MAX_T = 500
INTERVAL = 50
LEARNING_RATE = 1e-3  # or 0.0005 // 5e-3, 1e-3, 5e-4, e-4
REPLAY_BUFFER_SIZE = int(1e5)
EPS = 0.8
EPS_END = 0.05
EPS_DECAY = 0.95
EXPLORATION_STEPS = 64
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_NETWORK_UPDATE_FREQUENCY = 4
RANDMIN_BATCH_SIZE = 64  # 2, 4, 8, 16, 32
MAXMIN_N = 8  # 2, 4, 6, 8
AVG_N = 2  # 2, 4, 6, 8
VAR = 0


def get_agent(
    agent_name,
    env,
    state_dim,
    hidden_dim,
    action_dim,
    seed,
    lr,
    eps,
    eps_end,
    eps_decay,
    rbs,
    bs,
    gamma,
    tnuf,
    maxminn,
    avgn,
    randminbs,
):
    if agent_name == "dqn":
        agent = DQN(
            env,
            state_dim,
            hidden_dim,
            action_dim,
            seed,
            lr,
            eps,
            eps_end,
            eps_decay,
            rbs,
            bs,
            gamma,
            tnuf,
        )
    elif agent_name == "ddqn":
        agent = DDQN(
            env,
            state_dim,
            hidden_dim,
            action_dim,
            seed,
            lr,
            eps,
            eps_end,
            eps_decay,
            rbs,
            bs,
            gamma,
            tnuf,
        )
    elif agent_name == "maxmin":
        agent = MaxminDQN(
            env,
            state_dim,
            hidden_dim,
            action_dim,
            seed,
            maxminn,
            lr,
            eps,
            eps_end,
            eps_decay,
            rbs,
            bs,
            gamma,
            tnuf,
        )
    elif agent_name == "avg":
        agent = AveragedDQN(
            env,
            state_dim,
            hidden_dim,
            action_dim,
            seed,
            avgn,
            lr,
            eps,
            eps_end,
            eps_decay,
            rbs,
            bs,
            gamma,
            tnuf,
        )
    elif agent_name == "randmin":
        agent = RandMin(
            env,
            state_dim,
            hidden_dim,
            action_dim,
            seed,
            randminbs,
            lr,
            eps,
            eps_end,
            eps_decay,
            rbs,
            bs,
            gamma,
            tnuf,
        )
    return agent


def set_seed(env, seed):
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "--env", type=str, default="mountaincar")
    parser.add_argument(
        "--agent",
        "--agent",
        type=str,
        default="my3",
        choices=["dqn", "ddqn", "maxmin", "avg", "randmin"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--var", "--var", type=int, default=VAR)
    parser.add_argument("--n_ep", "--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--maxt", "--maxt", type=int, default=MAX_T)
    parser.add_argument("--itv", "--interval", type=int, default=INTERVAL)
    parser.add_argument("--lr", "--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--eps", "--eps", type=float, default=EPS)
    parser.add_argument("--eps_end", "--eps_end", type=float, default=EPS_END)
    parser.add_argument("--eps_decay", "--eps_decay", type=float, default=EPS_DECAY)
    parser.add_argument(
        "--rbs", "--replay_buffer_size", type=int, default=REPLAY_BUFFER_SIZE
    )
    parser.add_argument("--bs", "--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", "--gamma", type=float, default=GAMMA)
    parser.add_argument(
        "--tnuf",
        "--target_update_frequency",
        type=int,
        default=TARGET_NETWORK_UPDATE_FREQUENCY,
    )
    parser.add_argument(
        "--bs1", "--randmin_batch_size", type=int, default=RANDMIN_BATCH_SIZE
    )
    parser.add_argument("--n", "--maxmin_n", type=int, default=MAXMIN_N)
    parser.add_argument("--avg_n", "--avg_n", type=int, default=AVG_N)
    parser.add_argument("--n_runs", "--n_runs", type=int, default=N_RUNS)
    parser.add_argument("--seed", "--seed", type=int, default=0)
    args = parser.parse_args()

    if args.env == "cartpole":
        env = RandCartPoleEnv(variance=args.var)
        hidden_dim = 64
    elif args.env == "mountaincar":
        env = RandMountainCarEnv(variance=args.var)
        hidden_dim = 64
    elif args.env == "acrobot":
        env = RandAcrobotEnv(variance=args.var)
        hidden_dim = 64
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    rewards = np.empty((args.n_runs, args.n_ep))
    steps = np.empty((args.n_runs, args.n_ep))
    for i_run in range(args.n_runs):
        set_seed(env, i_run)
        agent = get_agent(
            args.agent,
            env,
            state_dim,
            hidden_dim,
            action_dim,
            args.seed,
            args.lr,
            args.eps,
            args.eps_end,
            args.eps_decay,
            args.rbs,
            args.bs,
            args.gamma,
            args.tnuf,
            args.n,
            args.avg_n,
            args.bs1,
        )
        cur_reward, cur_step = agent.train(args.n_ep, args.maxt, args.itv, args.verbose)
        rewards[i_run] = cur_reward
        steps[i_run] = cur_step

    np.save(os.path.join("result", env.name, agent.name + "_step.npy"), steps)
    np.save(os.path.join("result", env.name, agent.name + "_reward.npy"), rewards)

    f = open(os.path.join("result", env.name, agent.name + ".txt"), "a")
    f.write(str(args) + "\n")
    f.write(
        "steps {:.2f} $\\pm$ {:.2f}\t reward {:.2f} $\\pm$ {:.2f}\n".format(
            np.mean(steps[:, -1]),
            np.std(steps[:, -1]),
            np.mean(rewards[:, -1]),
            np.std(rewards[:, -1]),
        )
    )
    f.close()


if __name__ == "__main__":
    main()
