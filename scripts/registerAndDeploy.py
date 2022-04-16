import os, json, datetime, sys, argparse
from azureml.core import Workspace, Environment, Run

from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment

# +
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="input", default="./",)
args = parser.parse_args()

print("Argument 1: %s" % args.input)
# -

run = Run.get_context()
workspace = run.experiment.workspace

# +
with open(os.path.join(args.input, 'metric.json')) as f:
    metric = json.load(f)

model_path = os.path.join(args.input, './models')
model = Model.register(model_path = model_path, # this points to a local file
                       model_name = "arima-model-v2", # this is the name the model is registered as
                       tags = metric,
                       description = "arima-model-v2",
                       workspace = workspace)
# -

myenv = Environment.from_conda_specification(name="arima-env", file_path="./arima-env.yml")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=3, 
                                               tags={'name':'arima-inference-v2', 'framework': 'statsmodels'},
                                               description='arima inference v2')

service = Model.deploy(workspace=workspace,
                           name='arima-inference-v2', 
                           models=[model], 
                           inference_config=inference_config, 
                           deployment_config=aciconfig, overwrite=True)

service.wait_for_deployment(True)
print('Service Statet:')
print(service.state)
print('Service URI:')
print(service.scoring_uri)

print('Smoke Test:')
step_size=[3]
test_sample = json.dumps({"data": step_size})
test_sample = bytes(test_sample, encoding="utf8")
print(test_sample)

prediction = service.run(input_data=test_sample)
print(prediction)
