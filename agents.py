from mesa import Agent
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from commonVariable import *
from math import floor
from utilsUCS import PriorityQueue


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
        '''self.flock_num = 3
        # range in which other birds have to be present to the group to be considered a flock
        self.flock_area = 6'''
        self.sight = 7

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

    '''def flocking(self):
        flocking = False
        agents = [agent for agent in
                  self.model.grid.get_neighbors(self.pos, include_center=True, radius=self.flock_area, moore=True)
                  if isinstance(agent, type(self))]
        if len(agents) >= self.flock_num:
            flocking = True
        return flocking'''


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
        elif algorithm == "UCS":
            self.enemy = BirdAgentUCS
        elif algorithm == "GA":
            self.enemy = BirdAgentGA
        elif algorithm == "DRL":
            self.enemy = BirdAgentRL
        self.score = 0
        self.sight = 5
        self.turn_speed = 1
        # self.acceleration = 0.25
        # self.max_speed = 1
        # self.min_speed = 0.25

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
                if self.algorithm != "GA" and self.algorithm != "DRL":
                    bird[0].alive = False
                    self.model.grid.remove_agent(bird[0])
                else:
                    self.model.grid.move_agent(bird[0], (random.randint(0, self.model.grid.width-1),
                                               random.randint(0, self.model.grid.height-1)))

    def move(self):
        x, y = self.pos
        birds = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                  moore=True) if isinstance(agent, self.enemy)]
        #hanging_birds = [bird for bird in birds if not bird.flocking()]
        hanging_birds = birds
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


# q_bin = 6
o_bin = 8
d_bin = 5


class BirdAgentGA(BirdAgent):
    def __init__(self, unique_id, model, dna):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        self.speed = 0.5
        self.score = 0
        self.alive = True
        # self.occurred = []
        self.strategy = [None] * (np.power(o_bin, 2) * np.power(d_bin, 2))
        if dna is None:
            self.strategy = [random.randint(0, 7) for _ in self.strategy]
        else:
            self.strategy = dna

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
        '''q1b = np.clip(round(self.q_count((self.pos[0] - radius, self.pos[1] + radius), BirdAgentGA)), 0, q_bin-1)
        q2b = np.clip(round(self.q_count((self.pos[0] + radius, self.pos[1] + radius), BirdAgentGA)), 0, q_bin-1)
        q3b = np.clip(round(self.q_count((self.pos[0] + radius, self.pos[1] - radius), BirdAgentGA)), 0, q_bin-1)
        q4b = np.clip(round(self.q_count((self.pos[0] - radius, self.pos[1] - radius), BirdAgentGA)), 0, q_bin-1)'''
        index = fo + fd * o_bin + po * d_bin * o_bin + pd * d_bin * np.power(o_bin, 2)
        # self.occurred.append(int(index))
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

    '''def q_count(self, pos, object_type):
        found = len([agent for agent
                     in
                     self.model.grid.get_neighbors(pos, include_center=True, radius=floor(self.sight / 2), moore=False)
                     if isinstance(agent, object_type)])
        return found'''

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


class BirdAgentUCS(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        self.speed = 0.5
        self.score = 0
        self.turn_speed = 1
        self.acceleration = 0.25
        self.max_speed = 0.75
        self.min_speed = 0.25
        self.alive = True
        self.queue = None
        self.sight = 5

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
        tick = self.model.schedule.time
        inverse_speed = 1/self.speed
        if tick % inverse_speed == 0:
            x, y = self.pos
            z = self.ucs()
            best_move = z[1][0]

            self.model.grid.move_agent(self, (x+best_move[0], y+best_move[1]))

    def ucs(self):
        self.queue = PriorityQueue()
        self.queue.update((self.pos, [], self.pos_value(self.pos)), -self.pos_value(self.pos))
        visitedNodes = []
        while 1:
            if self.queue.isEmpty():
                break
            state = self.queue.pop()
            if state[0] not in visitedNodes:
                visitedNodes.append(state[0])
                if np.absolute(self.pos[0] - state[0][0]) == self.sight or \
                        np.absolute(self.pos[1] - state[0][1]) == self.sight:
                    return state

                if len(state[1]) < 2*self.sight:

                    successor_list = []
                    for x, y in direction_vector_list:
                        element = ((state[0][0] + x)%self.model.grid.width, (state[0][1] + y) % self.model.grid.height)
                        if np.absolute(self.pos[0] - element[0]) <= self.sight and np.absolute(self.pos[1] - element[1]) <= self.sight:
                            successor_list.append((element, (x, y), self.pos_value(element)))

                    for child in successor_list:
                        self.queue.update((child[0], state[1] + [child[1]], state[2]+child[2] - 1), -state[2]-child[2] + 1)

        return ((self.pos[0], self.pos[1] + self.sight), [(0, 1)] * self.sight,
                np.sum([self.pos_value(x) - 1 for x in [(self.pos[0], self.pos[1] + y)
                                                        for y in range(self.sight)]]))

    def pos_value(self, pos):
        pos = (np.mod(pos[0], self.model.grid.width), np.mod(pos[1], self.model.grid.height))
        objects = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                    moore=True) if not isinstance(agent, BirdAgentUCS)]

        value = 0
        for object in objects:
            if isinstance(object, PredatorAgent):
                distance = self.get_distance_with_wraparound(pos, object.pos)
                if distance == 0.0:
                    value -= 3 * self.model.score_for_death
                else:
                    value -= 1 / distance * self.model.score_for_death

            if isinstance(object, FoodAgent):
                distance = self.get_distance_with_wraparound(pos, object.pos)
                if distance == 0.0:
                    value += 3 * self.model.score_for_food
                else:
                    value += 1 / distance * self.model.score_for_food

        return value

    def get_distance_with_wraparound(self, p1, p2):
        p1 = (np.mod(p1[0], self.model.grid.width), np.mod(p1[1], self.model.grid.height))
        p2 = (np.mod(p2[0], self.model.grid.width), np.mod(p2[1], self.model.grid.height))
        min_dist = np.abs(np.array(p1) - np.array(p2))  # min_dist[0] = min x distance, min_dist[1] = min y distance
        # x
        if p1[0] < p2[0] and np.abs((p1[0]+self.model.grid.width) - p2[0]) < min_dist[0]:
            min_dist[0] = np.abs((p1[0]+self.model.grid.width) - p2[0])
        if p1[0] > p2[0] and np.abs(p1[1] - (p2[1]+self.model.grid.height)) < min_dist[0]:
            min_dist[0] = np.abs(p1[1] - (p2[1]+self.model.grid.height))
        # y
        if p1[1] < p2[1] and np.abs((p1[1]+self.model.grid.height) - p2[1]) < min_dist[1]:
            min_dist[1] = np.abs((p1[1]+self.model.grid.height) - p2[1])
        if p1[1] > p2[1] and np.abs(p1[1] - (p2[1]+self.model.grid.height)) < min_dist[1]:
            min_dist[1] = np.abs(p1[1] - (p2[1]+self.model.grid.height))
        return np.sqrt(np.power(min_dist[0], 2) + np.power(min_dist[1], 2))


from collections import deque
import time
import random
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 64
MODEL_NAME = "256x2"
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

epsilon = 1
EPSILON_DECAY = 0.975
MIN_EPSILON = 0.001


from utilsRL import ModifiedTensorBoard

class BirdAgentRL(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        self.speed = 0.5
        self.score = 0
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.alive = True
        self.queue = None
        self.sight = 7
        self.neuralNetwork = self.makeModel()
        self.targetNetwork = self.makeModel()
        self.targetNetwork.set_weights(self.neuralNetwork.get_weights())
        self.epsilon = epsilon
        self.current_state = None
        self.new_state = None
        
        self.tensorBoard = ModifiedTensorBoard(log_dir= f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.scoreresetcounter=0


    def makeModel(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(8, activation=tf.nn.softmax)])
        optimizerAdam  = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss="mse", optimizer=optimizerAdam, metrics=['accuracy'])
        #model = tf.keras.Sequential()
       # model.add(Conv2D(64, (5, 5), input_shape=(15, 15, 3)))
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.15))
        #model.add(Conv2D(128, (3, 3)))
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.15))
        #model.add(Flatten())
        #model.add(Dense(64, activation='relu'))
        #model.add(Dense(8)) # ACTION_SPACE_SIZE = how many choices (9)
        #model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def step(self):
        if self.alive:
            self.move()
            self.eat()
            self.scoreresetcounter = self.scoreresetcounter + 1
            if self.scoreresetcounter > 100:
                self.scoreresetcounter = 0
                self.score = 0
                print("RESETING AVERAGES")
            

    def dataFinder1(self):
        image = np.zeros((15, 15, 3))
        agents = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=True, radius=self.sight,
                                                                   moore=True) if not agent == self]
        for agent in agents:
            relative_pos = self.get_relative_position_with_wraparound(agent.pos, self.pos)
            if isinstance(agent, FoodAgent):
                image[relative_pos[0], relative_pos[1], 0] = image[relative_pos[0], relative_pos[1], 0] + 1
            if isinstance(agent, PredatorAgent):
                image[relative_pos[0], relative_pos[1], 1] = image[relative_pos[0], relative_pos[1], 1] + 1
            if isinstance(agent, BirdAgentRL):
                image[relative_pos[0], relative_pos[1], 2] = image[relative_pos[0], relative_pos[1], 2] + 1
        return image

 

    def get_relative_position_with_wraparound(self, p1, p2):
        p1 = np.array((np.mod(p1[0], self.model.grid.width), np.mod(p1[1], self.model.grid.height)))
        p2 = np.array((np.mod(p2[0], self.model.grid.width), np.mod(p2[1], self.model.grid.height)))
        min_dist = np.array(p1) - np.array(p2)  # min_dist[0] = min x distance, min_dist[1] = min y distance
        # x
        if p1[0] < p2[0] and np.abs((p1[0] + self.model.grid.width) - p2[0]) < np.abs(min_dist[0]):
            min_dist[0] = (p1[0] + self.model.grid.width) - p2[0]
        if p1[0] > p2[0] and np.abs(p1[0] - (p2[0] + self.model.grid.width)) < np.abs(min_dist[0]):
            min_dist[0] = p1[0] - (p2[0] + self.model.grid.width)
        # y
        if p1[1] < p2[1] and np.abs((p1[1] + self.model.grid.height) - p2[1]) < np.abs(min_dist[1]):
            min_dist[1] = (p1[1] + self.model.grid.height) - p2[1]
        if p1[1] > p2[1] and np.abs(p1[1] - (p2[1] + self.model.grid.height)) < np.abs(min_dist[1]):
            min_dist[1] = p1[1] - (p2[1] + self.model.grid.height)
        return min_dist + np.array((self.sight, self.sight))
            

    def eat(self):
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        if this_cell:
            grass = [obj for obj in this_cell if isinstance(obj, FoodAgent)]
            if grass:
                self.model.grid.remove_agent(grass[0])
                self.score += self.model.score_for_food

    def move(self):
        tick = self.model.schedule.time
        inverse_speed = 1/self.speed
        if tick % inverse_speed == 0:
            x, y = self.pos

            max_index = 0
            self.current_state = self.dataFinder(self.pos)
            if np.random.random() > self.epsilon:
            # Get action from Q table
                max_index = np.argmax(self.rl(self.current_state))
            else:
                # Get random action
                max_index = np.random.randint(0, 8)

            if max_index == 0:
                best_move = [0,-1]
            elif max_index == 1:
                best_move = [0,1]
            elif max_index == 2:
                best_move = [1,0]
            elif max_index == 3:
                best_move = [-1,0]
            elif max_index == 4:
                best_move = [1,-1]
            elif max_index == 5:
                best_move = [-1,-1]
            elif max_index == 6:
                best_move = [-1,1]
            elif max_index == 7:
                best_move = [1,1]  

            self.model.grid.move_agent(self, (x+best_move[0], y+best_move[1]))
            self.new_state = self.dataFinder(self.pos)
            self.update_replay_memory((self.current_state, max_index, self.calculateReward(), self.new_state))
            self.train(False)

            if self.epsilon > MIN_EPSILON:
                self.epsilon *= EPSILON_DECAY
                self.epsilon = max(MIN_EPSILON, self.epsilon)

    def calculateReward(self):
        reward = -1
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        if this_cell:
            grass = [obj for obj in this_cell if isinstance(obj, FoodAgent)]
            if grass:
                reward = self.model.score_for_food
            predator = [obj for obj in this_cell if isinstance(obj, PredatorAgent)]
            if predator:
                reward = self.model.score_for_death
        
        return reward
    

    def rl(self, state):

        prediction = self.neuralNetwork.predict([state])
        return prediction

    def dataFinder(self, pos):
        pos = (np.mod(pos[0], self.model.grid.width), np.mod(pos[1], self.model.grid.height))
        objects = [agent for agent in self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.sight,
                                                                    moore=True) if not isinstance(agent, BirdAgentRL)]
        rlData = [9999, -1, 9999, -1]
        #The Data for the RL model. Predator Distance, Predator Bearing, Food Distance, Food Bearing          
        for object in objects:
            if isinstance(object, PredatorAgent):
                distance = self.get_distance_with_wraparound(pos, object.pos)
                if rlData[0] > distance:
                    rlData[0] = distance
                    dir_vec = np.array(object.pos) - np.array(self.pos)
                    dir_vec = dir_vec / max(abs(dir_vec))
                    bearing = np.argmin([np.linalg.norm(similarity) for similarity in [np.array(bearing) - dir_vec for bearing in direction_vector_list]])
                    rlData[1] = bearing

            if isinstance(object, FoodAgent):
                distance = self.get_distance_with_wraparound(pos, object.pos)
                if rlData[2] > distance:
                    rlData[2] = distance
                    dir_vec = np.array(object.pos) - np.array(self.pos)
                    dir_vec = dir_vec / max(abs(dir_vec))
                    bearing = np.argmin([np.linalg.norm(similarity) for similarity in [np.array(bearing) - dir_vec for bearing in direction_vector_list]])
                    rlData[3] = bearing
        return rlData
    

    def get_distance_with_wraparound(self, p1, p2):
        p1 = (np.mod(p1[0], self.model.grid.width), np.mod(p1[1], self.model.grid.height))
        p2 = (np.mod(p2[0], self.model.grid.width), np.mod(p2[1], self.model.grid.height))
        min_dist = np.abs(np.array(p1) - np.array(p2))  # min_dist[0] = min x distance, min_dist[1] = min y distance
        # x
        if p1[0] < p2[0] and np.abs((p1[0]+self.model.grid.width) - p2[0]) < min_dist[0]:
            min_dist[0] = np.abs((p1[0]+self.model.grid.width) - p2[0])
        if p1[0] > p2[0] and np.abs(p1[1] - (p2[1]+self.model.grid.height)) < min_dist[0]:
            min_dist[0] = np.abs(p1[1] - (p2[1]+self.model.grid.height))
        # y
        if p1[1] < p2[1] and np.abs((p1[1]+self.model.grid.height) - p2[1]) < min_dist[1]:
            min_dist[1] = np.abs((p1[1]+self.model.grid.height) - p2[1])
        if p1[1] > p2[1] and np.abs(p1[1] - (p2[1]+self.model.grid.height)) < min_dist[1]:
            min_dist[1] = np.abs(p1[1] - (p2[1]+self.model.grid.height))
        return np.sqrt(np.power(min_dist[0], 2) + np.power(min_dist[1], 2))

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self,terminal_state):
        if len(self.replay_memory)< MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.neuralNetwork.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.targetNetwork.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
      
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q


            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.neuralNetwork.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, 
        verbose=0, shuffle=False, callbacks=[self.tensorBoard] if terminal_state else None)

        # Update target network counter every episode
        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.targetNetwork.set_weights(self.neuralNetwork.get_weights())
            self.target_update_counter = 0


