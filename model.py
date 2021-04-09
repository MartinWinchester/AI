from mesa import Agent, Model
import numpy as np
import random
from mesa.time import RandomActivation as Activation
from mesa.space import MultiGrid as Space
from mesa.datacollection import DataCollector


direction_dict = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1), "S": (0, -1), "SW": (-1, -1), "W": (-1, 0),
                  "NW": (-1, +1)}
directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def total_score(model):
    agent_score = sum([agent.score for agent in model.schedule.agents])
    return agent_score


class BirdAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        self.speed = random.randint(1, 4)/4
        self.score = 0
        self.turn_speed = 1
        self.acceleration = 0.25
        self.max_speed = 1
        self.min_speed = 0.25
        self.alive = True

    def step(self):
        if self.alive:
            self.move()
            self.eat()

    def eat(self):
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        if this_cell:
            grass = [obj for obj in this_cell if isinstance(obj, FoodAgent)]
            if grass:
                self.model.grid.remove_agent(grass[0])
                self.score += self.model.score_for_food
                print(str(self.unique_id) + " " + str(self.score))

    def move(self):
        if self.speed == self.min_speed:
            # if at min speed 50-50 accelerate or hold speed
            if random.uniform(0, 1) > 0.5:
                self.speed += self.acceleration
        elif self.speed == self.max_speed:
            # if at max speed 50-50 decelerate or hold speed
            if random.uniform(0, 1) > 0.5:
                self.speed -= self.acceleration
        else:
            # 33-33-33 accelerate, decelerate or hold speed
            rand = random.uniform(0, 1)
            if rand < 0.33:
                self.speed += self.acceleration
            elif rand > 0.66:
                self.speed -= self.acceleration
        tick = self.model.schedule.time
        inverse_speed = 1/self.speed
        if tick % inverse_speed == 0:
            x, y = self.pos
            curr_index = directions.index(self.orientation)
            valid_directions = []
            for i in range(-self.turn_speed, self.turn_speed+1):
                valid_directions.append(directions[(curr_index+i)%len(directions)])
            self.orientation = random.choice(valid_directions)
            self.model.grid.move_agent(self, (x+direction_dict[self.orientation][0], y+direction_dict[self.orientation][1]))


class FoodAgent(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class BirdModel(Model):
    def __init__(self, n, width, height, growth_wait=2):
        self.score_for_food = 10
        self.score_for_death = 100
        self.num_agents = n
        self.num_predator = round(n/4)
        self.growth_time = growth_wait
        self.grid = Space(width, height, True)
        self.schedule = Activation(self)
        self.max_food_id = 0
        self.running = True

        for i in range(self.num_agents):
            bird = BirdAgent(i, self)
            self.schedule.add(bird)
            x = random.randint(0, self.grid.width-1)
            y = random.randint(0, self.grid.height-1)
            self.grid.place_agent(bird, (x, y))

        for j in range(i + 1, i + 1 + self.num_predator):
            predator = PredatorAgent(j, self)
            self.schedule.add(predator)
            x = random.randint(0, self.grid.width - 1)
            y = random.randint(0, self.grid.height - 1)
            self.grid.place_agent(predator, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"TotalScore": total_score},
            agent_reporters={"Score": "score"})

    def step(self):
        if self.schedule.time % self.growth_time == 0:
            self.max_food_id += 1
            food = FoodAgent(self.max_food_id, self)
            x = random.randint(0, self.grid.width-1)
            y = random.randint(0, self.grid.height-1)
            self.grid.place_agent(food, (x, y))
        self.datacollector.collect(self)
        self.schedule.step()


class PredatorAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        self.speed = random.randint(1, 4) / 4
        self.score = 0
        self.turn_speed = 1
        self.acceleration = 0.25
        self.max_speed = 1
        self.min_speed = 0.25

    def step(self):
        self.move()
        self.eat()

    def eat(self):
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        if this_cell:
            bird = [obj for obj in this_cell if isinstance(obj, BirdAgent)]
            if bird:
                bird[0].alive = False
                bird[0].score -= self.model.score_for_death
                self.model.grid.remove_agent(bird[0])

    def move(self):
        if self.speed == self.min_speed:
            # if at min speed 50-50 accelerate or hold speed
            if random.uniform(0, 1) > 0.5:
                self.speed += self.acceleration
        elif self.speed == self.max_speed:
            # if at max speed 50-50 decelerate or hold speed
            if random.uniform(0, 1) > 0.5:
                self.speed -= self.acceleration
        else:
            # 33-33-33 accelerate, decelerate or hold speed
            rand = random.uniform(0, 1)
            if rand < 0.33:
                self.speed += self.acceleration
            elif rand > 0.66:
                self.speed -= self.acceleration
        tick = self.model.schedule.time
        inverse_speed = 1 / self.speed
        if tick % inverse_speed == 0:
            x, y = self.pos
            curr_index = directions.index(self.orientation)
            valid_directions = []
            for i in range(-self.turn_speed, self.turn_speed + 1):
                valid_directions.append(directions[(curr_index + i) % len(directions)])
            self.orientation = random.choice(valid_directions)
            self.model.grid.move_agent(self, (
            x + direction_dict[self.orientation][0], y + direction_dict[self.orientation][1]))