from datetime import datetime
import json
import random
from typing import Optional

from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import requests
import sqlite3

from model_scoring import load_model,run_scoring

app = FastAPI()

# Accepts only the following inputs from UI
class InputData(BaseModel):
    experiment_name: Optional[str]
    score_model_id: str
    scoreset: Optional[list] = None

class Experiment():
    score_model_id = None
    scoreset = None
    score = None
    experiment_name = None

ex = Experiment()

def assign_values(ip: InputData):
    ex.score_model_id = ip.score_model_id
    ex.scoreset = ip.scoreset
    ex.experiment_name = ip.experiment_name

def store_to_db():
    experiment_id = str(random.randint(1000,9999))
    insert_ts = str(datetime.now())

    insert_query = """INSERT INTO experiments (experiment_id, score_model_id, experiment_name, inputs, predictions, insert_ts) VALUES (?,?,?,?,?,?);"""
    insert_data = (experiment_id, ex.score_model_id, ex.experiment_name, str(ex.scoreset), str(ex.score), insert_ts)

    cur.execute(insert_query, insert_data)
    con.commit()
    return experiment_id

@app.on_event('startup')
async def on_startup():
    global con, cur
    con = sqlite3.connect('rtml.db', check_same_thread=False)
    cur = con.cursor()

@app.on_event('shutdown')
async def on_startup():
    con.close()

@app.get("/")
def root():
    return "RTML"


@app.get("/models", tags=["model"])
def show_models():
    with open("model_list.json","rb") as f:
        model_list = json.load(f)
    return model_list

@app.get("/models/data/{score_model_id}", tags=["model"])
def return_scoreset(score_model_id: str):
    with open(f"{score_model_id}.json","rb") as f:
        scoreset = json.load(f)
    print(scoreset)
    return scoreset

@app.post("/models/execute", tags=["model"])
def execute(input_data: InputData):
    assign_values(input_data)
    
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }

    url = 'http://127.0.0.1:9000/predict'
    data = "{\"data\":{\"ndarray\":[[\""+ex.score_model_id+"\"],"+str(ex.scoreset)+"]}}"

    response = requests.post(url=url, headers=headers, data=data, verify=False)

    prediction_proba = response.json()["data"]["ndarray"]
    ex.score = prediction_proba
    experiment_id = store_to_db()
    with open(f"{ex.score_model_id}.json", "rb") as f:
        scoreset = json.load(f)

    return_json = {"experiment_id": experiment_id, "target_class_names": scoreset["target_class_names"] ,"score": prediction_proba}
    return return_json

@app.get("/models/experiments", tags=["model"])
def show_experiments():
    res = cur.execute("SELECT * FROM experiments")
    res_rows = []
    for row in res:
        res_rows.append(row)
    return res_rows
