from mesa import Agent
import random
import numpy as np
from commonVariable import *
from math import floor
from utilsUCS import PriorityQueue
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time


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
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.orientation = random.choice(directions)
        # self.speed = random.randint(1, 4) / 4
        self.speed = 1
        self.enemy = None
        if self.model.algorithm == "Dummy":
            self.enemy = BirdAgent
        elif self.model.algorithm == "UCS":
            self.enemy = BirdAgentUCS
        elif self.model.algorithm == "GA":
            self.enemy = BirdAgentGA
        elif self.model.algorithm == "DRL":
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
                if self.model.algorithm != "GA" and self.model.algorithm != "DRL":
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
            #todo next should not be height and 1
        if p1[0] > p2[0] and np.abs(p1[1] - (p2[1]+self.model.grid.height)) < min_dist[0]:
            min_dist[0] = np.abs(p1[1] - (p2[1]+self.model.grid.height))
        # y
        if p1[1] < p2[1] and np.abs((p1[1]+self.model.grid.height) - p2[1]) < min_dist[1]:
            min_dist[1] = np.abs((p1[1]+self.model.grid.height) - p2[1])
        if p1[1] > p2[1] and np.abs(p1[1] - (p2[1]+self.model.grid.height)) < min_dist[1]:
            min_dist[1] = np.abs(p1[1] - (p2[1]+self.model.grid.height))
        return np.sqrt(np.power(min_dist[0], 2) + np.power(min_dist[1], 2))


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# An array with last n steps for training
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# noinspection SpellCheckingInspection
class BirdAgentRL(Agent):
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
        self.sight = 7
        # Main model
        self.this_model = self.create_model()
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.this_model.get_weights())
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.new_observation = None
        self.reward = 0
        self.done = False
        self.episode_step = 0
        self.last_score = 0

    def create_model(self):
        a = self.pos
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=(15, 15, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(8, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.this_model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.this_model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.this_model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.this_model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def get_image(self):
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

    def step(self):
        if self.alive:
            # This part stays mostly the same, the change is to query a model for Q values
            current_state = self.get_image()
            if np.random.random() > self.model.epsilon:
                # Get action from Q table
                action = np.argmax(self.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, 8)

            x, y = self.pos
            self.model.grid.move_agent(self, (x + direction_vector_list[action][0], y + direction_vector_list[action][1]))
            self.episode_step += 1
            #self.eat()

            new_state = self.get_image()

            # todo this doesnt look like its propagating the value right, tts almost always just -1
            # todo also there were to types of NNs that needed to be updated, am I updating both?
            score = self.score - self.last_score
            if score == 0:
                reward = -0.1
            else:
                reward = score

            done = False
            if self.episode_step >= 200:
                done = True
            self.update_replay_memory((current_state, action, reward, new_state, done))
            self.train(done)
