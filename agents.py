from mesa import Agent
import random
import numpy as np
from commonVariable import *
from math import floor


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
    def __init__(self, unique_id, model, algorithm):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        # self.speed = random.randint(1, 4) / 4
        self.speed = 1
        self.enemy = None
        self.algorithm = algorithm
        if algorithm == "Dummy":
            self.enemy = BirdAgent
        elif algorithm == "Q":
            raise NotImplementedError
        elif algorithm == "GA":
            self.enemy = BirdAgentGA
        elif algorithm == "DRL":
            raise NotImplementedError
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
            bird = [obj for obj in this_cell if isinstance(obj, self.enemy)]
            if bird:
                bird[0].score -= self.model.score_for_death
                if self.algorithm != "GA":
                    bird[0].alive = False
                    self.model.grid.remove_agent(bird[0])
                else:
                    self.model.grid.move_agent(bird[0], (random.randint(0, self.model.grid.width-1),
                                               random.randint(0, self.model.grid.height-1)))

    def move(self):
        x, y = self.pos
        birds = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                  moore=True) if isinstance(agent, self.enemy)]
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


q_bin = 6
o_bin = 8
d_bin = 5


class BirdAgentGA(Agent):
    def __init__(self, unique_id, model, dna):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        # self.speed = random.randint(1, 3) / 4
        self.speed = 0.5
        self.score = 0
        # self.turn_speed = 1
        # self.acceleration = 0.25
        # self.max_speed = 0.75
        # self.min_speed = 0.25
        self.alive = True
        self.strategy = [None] * 2073600
        if dna is None:
            self.strategy = [random.randint(0, 7) for _ in self.strategy]
        else:
            self.strategy = dna
        self.sight = 5

    def step(self):
        if self.alive:
            self.move()
            self.eat()

    def find_action(self):
        pd, po = self.object_dd(PredatorAgent)
        fd, fo = self.object_dd(FoodAgent)
        pd = np.clip(round(pd/2), 0, d_bin-1)
        fd = np.clip(round(fd/2), 0, d_bin-1)
        radius = floor(self.sight / 2)
        q1b = round(self.q_count((self.pos[0] - radius, self.pos[1] + radius), BirdAgentGA))
        q2b = round(self.q_count((self.pos[0] + radius, self.pos[1] + radius), BirdAgentGA))
        q3b = round(self.q_count((self.pos[0] + radius, self.pos[1] - radius), BirdAgentGA))
        q4b = round(self.q_count((self.pos[0] - radius, self.pos[1] - radius), BirdAgentGA))
        index = q4b + q3b * q_bin + q2b * np.power(q_bin, 2) + q1b * np.power(q_bin, 3) + fo * np.power(q_bin, 4) + fd \
               * o_bin * np.power(q_bin, 4) + po * d_bin * o_bin * np.power(q_bin, 4) + pd * d_bin * np.power(q_bin, 4) \
               * np.power(o_bin, 2)
        self.orientation = self.strategy[int(index)]

    def object_dd(self, object_type):
        objects = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                    moore=True) if isinstance(agent, object_type)]
        target = None
        target_dist = None
        if objects:
            for obj in objects:
                abs_distance_vector = np.absolute(np.array(self.pos) - np.array(obj.pos))
                distance = abs_distance_vector[0] + abs_distance_vector[1]
                if target:
                    if target_dist > distance:
                        target = obj
                        target_dist = distance
                else:
                    target = obj
                    target_dist = distance
            dir_vec = np.array(target.pos) - np.array(self.pos)
            dir_vec = dir_vec / max(abs(dir_vec))
            bearing = np.argmin([np.linalg.norm(similarity) for similarity in
                                 [np.array(bearing) - dir_vec for bearing in direction_vector_list]])
            return target_dist, bearing
        return 0, 0

    def q_count(self, pos, object_type):
        found = len([agent for agent
                     in
                     self.model.grid.get_neighbors(pos, include_center=True, radius=floor(self.sight / 2), moore=False)
                     if isinstance(agent, object_type)])
        return found

    def eat(self):
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        if this_cell:
            grass = [obj for obj in this_cell if isinstance(obj, FoodAgent)]
            if grass:
                self.model.grid.remove_agent(grass[0])
                self.score += self.model.score_for_food

    def move(self):
        tick = self.model.schedule.time
        inverse_speed = 1 / self.speed
        if tick % inverse_speed == 0:
            x, y = self.pos
            self.find_action()
            self.model.grid.move_agent(self, (x + direction_dict[directions[self.orientation]][0],
                                              y + direction_dict[directions[self.orientation]][1]))