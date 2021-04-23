from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from model import BirdModel
from agents import BirdAgent, FoodAgent, PredatorAgent, BirdAgentGA, BirdAgentUCS
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", default=20, help="population size")
parser.add_argument("-a", "--algorithm", default="Dummy", help="algorithm to be used, use one of Dummy, GA, Q, DRL")

args = parser.parse_args()


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5}

    if isinstance(agent, BirdAgent):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0
    if isinstance(agent, BirdAgentGA):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0
    elif isinstance(agent, FoodAgent):
        portrayal["Color"] = "green"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.3
    elif isinstance(agent, PredatorAgent):
        portrayal["Color"] = "black"
        portrayal["Layer"] = 2
        portrayal["r"] = 1
    elif isinstance(agent, BirdAgentUCS):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0
    return portrayal


width = 100
height = 50
grid = CanvasGrid(agent_portrayal, width, height, 800, 800*height/width)

chart = ChartModule([{"Label": "TotalScore",
                      "Color": "Black"}],
                    data_collector_name='dc')
if args.algorithm == "GA":
    with open("dnas.txt", "rb") as fp:
        dnas = pickle.load(fp)

    server = ModularServer(BirdModel,
                       [grid, chart],
                       "Bird Model",
                       {"n": int(args.number), "width": width, "height": height, "algorithm": args.algorithm, "dnas":dnas})

else:
    server = ModularServer(BirdModel,
                           [grid, chart],
                           "Bird Model",
                           {"n": int(args.number), "width": width, "height": height, "algorithm": args.algorithm})
server.port = 8521
server.launch()