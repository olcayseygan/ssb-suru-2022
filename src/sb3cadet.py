import argparse

import yaml
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from agents.agent import TrainAgentEnv


def read_hypers():
    with open(f"./src/hyper.yaml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict["agentsofglory"]


parser = argparse.ArgumentParser(description='Cadet Agents')
parser.add_argument('map', metavar='map', type=str,
                    help='Select Map to Train')
parser.add_argument('--mode', metavar='mode', type=str, default="Train",
                    help='Select Mode[Train,Sim]')
parser.add_argument('--agentBlue', metavar='agentBlue', type=str, default="RayEnv",
                    help='Class name of Blue Agent')
parser.add_argument('agentRed', metavar='agentRed', type=str,
                    help='Class name of Red Agent')
parser.add_argument('--numOfMatch', metavar='numOfMatch', type=int, nargs='?', default=10,
                    help='Number of matches to play between agents')
parser.add_argument('--render', action='store_true',
                    help='Render the game')
parser.add_argument('--gif', action='store_true',
                    help='Create a gif of the game, also sets render')
parser.add_argument('--img', action='store_true',
                    help='Save images of each turn, also sets render')

parser.add_argument("--version", type=str)
parser.add_argument('--prefix', type=str)

args = parser.parse_args()
agents = [None, args.agentRed]


if __name__ == "__main__":
    hyperparams = read_hypers()

    for agentsofglory in hyperparams:
        gamename, hyperparam = list(agentsofglory.items())[0]

        env = SubprocVecEnv([lambda: TrainAgentEnv(args, agents) for i in range(hyperparam["env"]["n_envs"])]) # type: ignore
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f'./models/{args.version}', name_prefix=args.prefix)
        model = A2C(env=env, verbose=1, tensorboard_log="logs", device="cuda", **hyperparam["agent"])
        model.learn(callback=[checkpoint_callback], tb_log_name=args.prefix, **hyperparam["learn"])
        model.save(f"./models/{args.prefix}")
