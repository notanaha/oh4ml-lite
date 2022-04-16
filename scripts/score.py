import pickle
import os, json
import numpy
from azureml.core.model import Model
from statsmodels.tsa.arima_model import ARIMA


def init():
    global model
    import joblib

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'models/arima_model.pkl')
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result=model.forecast(steps=data[0])[0]
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
