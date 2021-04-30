from model import BirdModel
from agents import BirdAgentGA, BirdAgentRL
import argparse
from utilsGA import *
import time
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", default=20, help="population size")
parser.add_argument("-l", "--load", default="False", help="load pre-trained model from file")
parser.add_argument("-a", "--algorithm", default="Dummy", help="algorithm to be used, use one of Dummy, GA, Q, DRL")
parser.add_argument("-s", "--steps", default=100, help="number of steps to run in a batch")
parser.add_argument("-lr", "--learning_rate", default=0.1, help="learning rate")
parser.add_argument("-cp", "--checkpoint", default=10, help="number of iterations to run before creating a checkpoint")
parser.add_argument("-m", "--mutation", default=0.02, help="how many time steps to use in one generation")
parser.add_argument("-p", "--predators", default=6, help="number of predators")
parser.add_argument("-f", "--food", default="True", help="if true add food to grid")

args = parser.parse_args()
start = time.perf_counter()
n = int(args.number)
if args.predators is not None:
    p = int(args.predators)
else:
    p = None
f = str(args.food).lower() == "true"
if args.algorithm.lower() == "ga":
    best_dna = []
    best_dna_score = -999999999
    bests = []
    worsts = []
    totals = []
    times = []
    if str(args.load).lower() == "true":
        with open("dnas.txt", "rb") as fp:
            dnas = pickle.load(fp)
            model = BirdModel(n, 100, 50, args.algorithm, dnas, predators=p, food=f)
    else:
        model = BirdModel(n, 100, 50, args.algorithm, predators=p, food=f)
    util = Utils()
    epoch = 1
    while 1:
        for i in range(int(args.steps)):
            model.step()
        agents = [agent for agent in model.schedule.agents if isinstance(agent, BirdAgentGA)]
        scores = [agent.score for agent in model.schedule.agents if isinstance(agent, BirdAgentGA)]
        fitness_tuples = calc_fitness(agents)
        if fitness_tuples[0][0] > best_dna_score:
            best_dna = fitness_tuples[0][1]
            best_dna_score = fitness_tuples[0][0]
        new_gen = util.natural_selection(fitness_tuples, n, float(args.mutation))
        print("Generation: " + str(epoch))
        epoch += 1
        total = model.dc.model_vars["TotalScore"][-1]
        best = max(scores)
        worst = min(scores)
        end = time.perf_counter()
        ellapsed_time = end-start
        totals.append(total)
        bests.append(best)
        worsts.append(worst)
        times.append(ellapsed_time)
        if np.mod(epoch, int(args.checkpoint)) == 0:
            with open("dnas.txt", "wb") as fp:
                pickle.dump([ag.strategy for score, ag in fitness_tuples], fp)
            with open("best_dna.txt", "wb") as fp:
                pickle.dump(best_dna, fp)
            print("Totals")
            for tot in totals:
                print(str(tot))
            print("Bests")
            for bes in bests:
                print(str(bes))
            print("Worsts")
            for wors in worsts:
                print(str(wors))
            print("Times")
            for tim in times:
                print(str(tim))
            df = pd.DataFrame(columns=['Total', 'Best', 'Worst', 'Time'])
            df['Total'] = totals
            df['Best'] = bests
            df['Worst'] = worsts
            df['Time'] = times
            df.to_csv("records.csv")
        start = time.perf_counter()
        del model
        model = BirdModel(n, 100, 50, args.algorithm, new_gen, predators=p, food=f)

if args.algorithm.lower() == "drl":
    bests = []
    worsts = []
    totals = []
    times = []
    epoch = 0
    steps = int(args.steps)

    # Learning rate here actually corresponds to epsilon decay
    model = BirdModel(n, 100, 50, args.algorithm, predators=p, food=f, lr=float(args.learning_rate))
    agents = [agent for agent in model.schedule.agents if isinstance(agent, BirdAgentRL)]
    while 1:
        for agent in agents:
            agent.score = 0
        start = time.perf_counter()
        for _ in range(0, steps):
            model.step()
        end = time.perf_counter()
        scores = [agent.score for agent in model.schedule.agents if isinstance(agent, BirdAgentRL)]
        total = model.dc.model_vars["TotalScore"][-1]
        best = max(scores)
        worst = min(scores)
        ellapsed_time = end-start
        totals.append(total)
        bests.append(best)
        worsts.append(worst)
        times.append(ellapsed_time)
        epoch += 1
        if np.mod(epoch, int(args.checkpoint)) == 0:
            print("Totals")
            for tot in totals:
                print(str(tot))
            print("Bests")
            for bes in bests:
                print(str(bes))
            print("Worsts")
            for wors in worsts:
                print(str(wors))
            print("Times")
            for tim in times:
                print(str(tim))

        # Decay epsilon
        if model.epsilon > model.min_epsilon:
            model.epsilon *= model.lr
            model.epsilon = max(model.min_epsilon, model.epsilon)


if args.algorithm.lower() != "ga" and args.algorithm.lower() != "drl":
    model = BirdModel(n, 100, 50, args.algorithm, predators=p, food=f)
    for i in range(int(args.steps)):
        model.step()
        print(model.dc.model_vars["TotalScore"][-1])
