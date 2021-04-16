from model import BirdModel
from agents import BirdAgentGA
import argparse
from utilsGA import *
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", default=20, help="population size")
parser.add_argument("-a", "--algorithm", default="Dummy", help="algorithm to be used, use one of Dummy, GA, Q, DRL")
parser.add_argument("-s", "--steps", default=100, help="number of steps to run in a batch")
parser.add_argument("-lr", "--learning_rate", default=0.1, help="learning rate")
parser.add_argument("-cp", "--checkpoint", default=10, help="number of iterations to run before creating a checkpoint")
parser.add_argument("-m", "--mutation", default=0.05, help="how many time steps to use in one generation")

args = parser.parse_args()
start = time.perf_counter()
n = int(args.number)
model = BirdModel(n, 100, 50, args.algorithm)

if args.algorithm == "GA":
    bests = []
    worsts = []
    totals = []
    times = []
    model = BirdModel(n, 100, 50, args.algorithm)
    util = Utils()
    iter = 1
    while 1:
        for i in range(int(args.steps)):
            model.step()
        agents = [agent for agent in model.schedule.agents if isinstance(agent, BirdAgentGA)]
        scores = [agent.score for agent in model.schedule.agents if isinstance(agent, BirdAgentGA)]
        fitness_tuples = calc_fitness(agents)
        new_gen = util.natural_selection(fitness_tuples, n, args.mutation)
        print("Generation: " + str(iter))
        iter += 1
        total = model.dc.model_vars["TotalScore"][-1]
        best = max(scores)
        worst = min(scores)
        # print("Total Score: " + str(total))
        # print("Best Score: " + str(best))
        # print("Worst Score: " + str(worst))
        end = time.perf_counter()
        ellapsed_time = end-start
        # print("Elapsed time: " + str(time))
        # save, log info
        totals.append(total)
        bests.append(best)
        worsts.append(worst)
        times.append(ellapsed_time)
        if np.mod(iter, args.checkpoint) == 0:
            with open("dnas.txt", "wb") as fp:
                pickle.dump([ag.strategy for score, ag in fitness_tuples], fp)
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
        start = time.perf_counter()
        model = BirdModel(n, 100, 50, args.algorithm, new_gen)


if args.algorithm != "GA":
    for i in range(int(args.steps)):
        model.step()
        print(model.dc.model_vars["TotalScore"][-1])
