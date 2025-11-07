import joblib
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import os

import sys
print("Starting model load...", file=sys.stderr)
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    print("Model loaded successfully.", file=sys.stderr)
except Exception as e:
    print(f"Model load error: {e}", file=sys.stderr)
    raise

class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>Iris model API</h2><a href='/form'>Try the web form</a>"

@app.get("/form", response_class=HTMLResponse)
def form():
    return """
    <form action='/form' method='post'>
      Sepal length: <input name='f1' type='number' step='any'><br>
      Sepal width: <input name='f2' type='number' step='any'><br>
      Petal length: <input name='f3' type='number' step='any'><br>
      Petal width: <input name='f4' type='number' step='any'><br>
      <input type='submit' value='Predict'>
    </form>
    """

@app.post("/form", response_class=HTMLResponse)
def predict_form(f1: float = Form(...), f2: float = Form(...), f3: float = Form(...), f4: float = Form(...)):
    features = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return f"<h2>Predicted class: {class_name}</h2><a href='/form'>Try again</a>"

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}