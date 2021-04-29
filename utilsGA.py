import numpy as np
import random

dna_size = 1600


def calc_fitness(agents):
    fitness_tuples = []
    for agent in agents:
        fitness_tuples.append((agent.score, agent))
    return sorted(fitness_tuples, key=lambda x: x[0], reverse=True)


class Utils:
    def __init__(self):
        self.mating_pool = []

    def natural_selection(self, fitness_tuples, num, mutation):
        self.mating_pool = []
        scores = [score for score, ag in fitness_tuples]
        minimum = np.min(scores)
        maximum = np.max(scores)
        difference = maximum - minimum
        if difference == 0:
            scores = [1 for score in scores]
        else:
            scores = [round(100 * (score - minimum)/difference) for score in scores]
        # hopefully non of this changes the order of elements
        for i in range(len(scores)):
            for j in range(int(scores[i])):
                self.mating_pool.append(fitness_tuples[i])
        return self.recombine(num, mutation)

    def recombine(self, num, mutation):
        new_gen = []
        for i in range(num):
            parent_1 = random.choice(self.mating_pool)[1]
            parent_2 = random.choice(self.mating_pool)[1]
            crossover_point = random.randint(dna_size/2-0.1*dna_size, dna_size/2+0.1*dna_size)
            child_dna = parent_1.strategy[:crossover_point] + parent_2.strategy[crossover_point:]
            mutate_gene_num = dna_size * mutation
            mutate_gene_num = random.randint(round(0.9*mutate_gene_num),
                                             round(1.1*mutate_gene_num))
            for i in range(mutate_gene_num):
                child_dna[random.randint(0, dna_size-1)] = random.randint(0, 7)
            '''for index in parent_2.occurred:
                child_dna[index] = parent_2.strategy[index]
            for index in parent_1.occurred:
                child_dna[index] = parent_1.strategy[index]'''
            new_gen.append(child_dna)
        return new_gen

    '''def natural_selection(self, fitness_tuples, num, mutation):
        self.mating_pool = []
        scores = [score for score, ag in fitness_tuples]
        minimum = np.min(scores)
        maximum = np.max(scores)
        difference = maximum - minimum
        if difference == 0:
            scores = [1 for _ in scores]
        else:
            scores = [round(100 * (score - minimum)/difference) for score in scores]
        new_gen = []
        for i in range(num):
            # sometimes chooses 0 score agents
            parents = np.random.choice([ag for _, ag in fitness_tuples], 2, scores)
            crossover_point = random.randint(dna_size/2-0.1*dna_size, dna_size/2+0.1*dna_size)
            child_dna = parents[0].strategy[:crossover_point] + parents[1].strategy[crossover_point:]
            mutate_gene_num = dna_size * mutation
            mutate_gene_num = random.randint(round(0.9*mutate_gene_num),
                                             round(1.1*mutate_gene_num))
            for i in range(mutate_gene_num):
                child_dna[random.randint(0, dna_size-1)] = random.randint(0, 7)
            new_gen.append(child_dna)
        return new_gen'''
