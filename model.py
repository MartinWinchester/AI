from mesa import Agent, Model
import random
import numpy as np
from mesa.time import RandomActivation as Activation
from mesa.space import MultiGrid as Space
from mesa.datacollection import DataCollector
from modelGA import BirdAgentGA

direction_dict = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1), "S": (0, -1), "SW": (-1, -1), "W": (-1, 0),
                  "NW": (-1, +1)}
direction_vector_list = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, +1)]
directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def total_score(model):
    agent_score = sum([agent.score for agent in model.schedule.agents])
    return agent_score


class BirdAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        self.speed = random.randint(1, 3)/4
        self.score = 0
        self.turn_speed = 1
        self.acceleration = 0.25
        self.max_speed = 0.75
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


class PredatorAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        # self.speed = random.randint(1, 4) / 4
        self.speed = 1
        self.score = 0
        self.sight = 5
        self.turn_speed = 1
        # self.acceleration = 0.25
        # self.max_speed = 1
        # self.min_speed = 0.25
        self.flock_num = 3
        # range in which other birds have to be present to the group to be considered a flock
        self.flock_area = 3

    def step(self):
        self.eat()
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
        x, y = self.pos
        birds = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                  moore=True) if isinstance(agent, BirdAgent)]
        hanging_birds = [bird for bird in birds if not self.is_in_flock(bird)]
        target = None
        target_dist = None
        if hanging_birds:
            for bird in hanging_birds:
                distance = np.linalg.norm(np.array(self.pos) - np.array(bird.pos))
                if target:
                    if target_dist > distance:
                        target = bird
                        target_dist = distance
                else:
                    target = bird
                    target_dist = distance
            dir_vec = np.array(target.pos) - np.array(self.pos)
            dir_vec = dir_vec / max(abs(dir_vec))
            bearing = direction_vector_list[np.argmin([np.linalg.norm(similarity) for similarity in
                                                       [np.array(bearing) - dir_vec for bearing in direction_vector_list]])]
            self.model.grid.move_agent(self, (x + bearing[0],
                                              y + bearing[1]))

        else:
            curr_index = directions.index(self.orientation)
            valid_directions = []
            for i in range(-self.turn_speed, self.turn_speed + 1):
                valid_directions.append(directions[(curr_index + i) % len(directions)])
            self.orientation = random.choice(valid_directions)
            self.model.grid.move_agent(self, (x + direction_dict[self.orientation][0],
                                              y + direction_dict[self.orientation][1]))

    def is_in_flock(self, bird):
        flocking = False
        if len([agent for agent
                in self.model.grid.get_neighbors(bird.pos, include_center=False, radius=self.flock_area, moore=True)
                if isinstance(agent, BirdAgent)]) > self.flock_num:
            flocking = True
        return flocking


class BirdModel(Model):
    def __init__(self, n, width, height, algorithm="Dummy"):
        self.score_for_food = 10
        self.score_for_death = 100
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
            elif algorithm == "Q":
                raise NotImplementedError
            elif algorithm == "GA":
                bird = BirdAgentGA(i, self)
            elif algorithm == "DRL":
                raise NotImplementedError
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
