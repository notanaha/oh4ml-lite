import os, json, datetime, sys, argparse
from azureml.core import Workspace, Environment, Run

from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="input", default="./",)
args = parser.parse_args()

print("Argument 1: %s" % args.input)


run = Run.get_context()
workspace = run.experiment.workspace


with open(os.path.join(args.input, 'metric.json')) as f:
    metric = json.load(f)


model_path = os.path.join(args.input, './models')
print(model_path)
print(metric['model_name'])

model = Model.register(model_path = model_path, # this points to a local file
                       model_name = metric['model_name'], # this is the name the model is registered as
                       tags = metric,
                       description = "arima model",
                       workspace = workspace)
