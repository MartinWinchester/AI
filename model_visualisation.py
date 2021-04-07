from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from model import BirdModel, BirdAgent, FoodAgent


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5}

    if isinstance(agent, BirdAgent):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0
    elif isinstance(agent, FoodAgent):
        portrayal["Color"] = "green"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.3
    return portrayal


n = 20
width = 100
height = 50
growth_wait = 2
grid = CanvasGrid(agent_portrayal, width, height, 800, 800*height/width)

server = ModularServer(BirdModel,
                       [grid],
                       "Bird Model",
                       {"n": n, "width": width, "height": height, "growth_wait": growth_wait})

server.port = 8521 # The default
server.launch()