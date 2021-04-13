from mesa import Agent
import random
from model import directions, FoodAgent, direction_dict,  direction_vector_list, PredatorAgent
import numpy as np
from math import floor


class BirdAgentGA(Agent):
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
        self.strategy = np.array(2073600)
        self.sight = 5

    def step(self):
        if self.alive:
            self.move()
            self.eat()

    def find_action(self):
        pd, po = self.object_dd(PredatorAgent)
        fd, fo = self.object_dd(FoodAgent)
        new_var = floor(self.sight/2)
        q1b = self.q_count((self.pos[0] - new_var, self.pos[1] + new_var), BirdAgentGA)
        q2b = self.q_count((self.pos[0] + new_var, self.pos[1] + new_var), BirdAgentGA)
        q3b = self.q_count((self.pos[0] + new_var, self.pos[1] - new_var), BirdAgentGA)
        q4b = self.q_count((self.pos[0] - new_var, self.pos[1] - new_var), BirdAgentGA)
        indx = q4b + q3b * 6 + q2b * 6 * 6 + q1b * 6 * 6 * 6 + fo * 6 * 6 * 6 * 6 + fd * 8 * 6 * 6 * 6 * 6 \
               + po * 5 * 8 * 6 * 6 * 6 * 6 + pd * 8 * 5 * 8 * 6 * 6 * 6 * 6
        self.orientation = self.strategy[indx]

    def object_dd(self, object_type):
        objects = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                  moore=True) if isinstance(agent, object_type)]
        target = None
        target_dist = None
        if objects:
            for object in objects:
                distance = np.linalg.norm(np.array(self.pos) - np.array(object.pos))
                if target:
                    if target_dist > distance:
                        target = object
                        target_dist = distance
                else:
                    target = object
                    target_dist = distance
            dir_vec = np.array(target.pos) - np.array(self.pos)
            dir_vec = dir_vec / max(abs(dir_vec))
            bearing = np.argmin([np.linalg.norm(similarity) for similarity in
                                 [np.array(bearing) - dir_vec for bearing in direction_vector_list]])
            return target_dist, bearing
        return 0, 0

    def q_count(self, pos, object_type):
        found = len([agent for agent
                in self.model.grid.get_neighbors(pos, include_center=True, radius=floor(self.sight/2), moore=False)
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

