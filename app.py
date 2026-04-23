from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

model = None
target_names = None


@app.on_event("startup")
def train_model():
    global model, target_names
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(sl: float, sw: float, pl: float, pw: float):
    features = [[sl, sw, pl, pw]]
    prediction = int(model.predict(features)[0])
    class_name = str(target_names[prediction])
    return {"prediction": prediction, "class_name": class_name}
