import os, json, datetime, sys, argparse
from azureml.core import Workspace, Environment, Run
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="input", default="./",)
args = parser.parse_args()


print("Argument 1: %s" % args.input)


with open(os.path.join(args.input, 'metric.json')) as f:
    metric = json.load(f)

    
run = Run.get_context()
workspace = run.experiment.workspace


model = Model(workspace, metric['model_name'])  #, version=metric['version'])
print('The model version is "' + str(model.version) + '"')

myenv = Environment.from_conda_specification(name="arima-env", file_path="./arima-env.yml")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=3, 
                                               tags={'name':metric['model_name'], 'framework': 'statsmodels'},
                                               description='arima inference')

service = Model.deploy(workspace=workspace,
                           name=metric['model_name'], 
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
