from mesa import Model
import random
from mesa.time import RandomActivation as Activation
from mesa.space import MultiGrid as Space
from mesa.datacollection import DataCollector
from agents import BirdAgent, PredatorAgent, FoodAgent, BirdAgentGA, BirdAgentUCS


def total_score(model):
    agent_score = sum([agent.score for agent in model.schedule.agents])
    return agent_score


class BirdModel(Model):
    def __init__(self, n, width, height, algorithm="Dummy", dnas=None):
        self.score_for_food = 10
        self.score_for_death = 20
        self.num_agents = n
        self.num_predator = round(n/4)
        self.growth_time = 2
        self.grid = Space(width, height, True)
        self.schedule = Activation(self)
        self.max_food_id = 0
        self.running = True
        for i in range(self.num_agents):
            if algorithm == "Dummy":
                bird = BirdAgent(i, self)
            elif algorithm == "UCS":
                bird = BirdAgentUCS(i, self)
            elif algorithm == "GA":
                if dnas is None:
                    bird = BirdAgentGA(i, self, dnas)
                else:
                    bird = BirdAgentGA(i, self, dnas[i])
            elif algorithm == "DRL":
                raise NotImplementedError
            self.schedule.add(bird)
            x = random.randint(0, self.grid.width-1)
            y = random.randint(0, self.grid.height-1)
            self.grid.place_agent(bird, (x, y))

        for j in range(i + 1, i + 1 + self.num_predator):
            predator = PredatorAgent(j, self, algorithm)
            self.schedule.add(predator)
            x = random.randint(0, self.grid.width - 1)
            y = random.randint(0, self.grid.height - 1)
            self.grid.place_agent(predator, (x, y))

        self.dc = DataCollector(
            model_reporters={"TotalScore": total_score},
            agent_reporters={"Score": "score"})

    def step(self):
        if self.schedule.time % self.growth_time == 0:
            self.max_food_id += 1
            food = FoodAgent(self.max_food_id, self)
            x = random.randint(0, self.grid.width-1)
            y = random.randint(0, self.grid.height-1)
            self.grid.place_agent(food, (x, y))
        self.dc.collect(self)
        self.schedule.step()
