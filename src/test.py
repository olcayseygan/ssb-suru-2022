
import argparse
import sys

import numpy as np
from gym import spaces

from agents.agent import EvaluationAgent
from game import Game

inputDirectory = '/data/inputs'
outputDirectory = '/data/outputs'
sys.path.append(inputDirectory)


class Evaluator():
    def __init__(self, args, agents):

        self.game = Game(args, agents)
        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(24*18*10+4,),
            dtype=np.int16
        )
        self.action_space = self.action_space = spaces.MultiDiscrete(
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5])
        self.agent = EvaluationAgent(self.observation_space, self.action_space)

    def evaluate(self, repetition=30):
        eps_rewards = []

        for _ in range(repetition):
            state = self.game.reset()
            done = False
            rewards = []
            while not done:
                action = self.agent.act(state)
                state, reward, done = self.game.step(action)
                rewards.append(reward)

            eps_rewards.append(sum(rewards))
        return np.mean(eps_rewards).item()


parser = argparse.ArgumentParser(description='Cadet Agents')

parser.add_argument('--mode', metavar='mode', type=str, default="Sim",
                    help='Select Mode[Train,Sim]')
parser.add_argument('--numOfMatch', metavar='numOfMatch', type=int, nargs='?', default=1,
                    help='Number of matches to play between agents')
parser.add_argument('--render', action='store_true',
                    help='Render the game')
parser.add_argument('--gif', action='store_true',
                    help='Create a gif of the game, also sets render')
parser.add_argument('--img', action='store_true',
                    help='Save images of each turn, also sets render')

args = parser.parse_args()

data = {
    'table': [],
    "results": [{}]
}
scores = []


for index, (map, agent) in enumerate([("TrainSingleTruckLarge", "EvaluationAgent")]):

    agents = [None, agent]
    args.map = map
    score = Evaluator(args, agents).evaluate(30)

    name = "RiskyValley" if index == 0 else f"SecretMap_{index}"
    score_name = f"{name}_score"

    data['table'].append(
        {
            'header': name,
            'field': score_name
        }
    )
    data["results"][0][score_name] = score
    scores.append(score)
data["total_score"] = sum(scores)

# with open(os.path.join(outputDirectory, "output.json"), 'x') as outfile:
#     json.dump(data, outfile)
