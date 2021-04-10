from model import BirdModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", default="Dummy", help="algorithm to be used")
parser.add_argument("-s", "--steps", default="100", help="number of steps to run in a batch")
parser.add_argument("-lr", "--learning_rate", default="100", help="learning rate")

args = parser.parse_args()

model = BirdModel(20, 100, 50, args.algorithm)

for i in range(int(args.steps)):
    model.step()
    print(model.dc.model_vars["TotalScore"][-1])