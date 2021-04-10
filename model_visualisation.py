from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from model import BirdModel, BirdAgent, FoodAgent, PredatorAgent


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
    elif isinstance(agent, PredatorAgent):
        portrayal["Color"] = "black"
        portrayal["Layer"] = 2
        portrayal["r"] = 1
    return portrayal


n = 20
width = 100
height = 50
grid = CanvasGrid(agent_portrayal, width, height, 800, 800*height/width)

chart = ChartModule([{"Label": "TotalScore",
                      "Color": "Black"}],
                    data_collector_name='dc')

server = ModularServer(BirdModel,
                       [grid, chart],
                       "Bird Model",
                       {"n": n, "width": width, "height": height})

server.port = 8521
server.launch()