"""
Программа: Модель для пердсказания того, будет ли человек трудоустроен в течение года.
Версия: 1.0
"""

import warnings
import optuna
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics
from src.clustering.clustering import pipeline_clustering


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH =  "../config/params.yml"


class Person(BaseModel):
    """
    Признаки для получения результатов модели
    """
    Female: int
    Tenure: float
    Birthyear: int
    Status: str
    Geography: str
    Province: str
    Matric: float
    Schoolquintile: float
    Diploma: float
    Math: str
    Mathlit: str
    Additional_lang: str
    Home_lang: str
    Degree: float
    Science: str
    Birthmonth: int

@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {'message': 'Hello'}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:100]}


@app.post("/predict_input")
def prediction_input(person: Person):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            person.Female,
            person.Tenure,
            person.Birthyear,
            person.Status,
            person.Geography,
            person.Province,
            person.Matric,
            person.Schoolquintile,
            person.Diploma,
            person.Math,
            person.Mathlit,
            person.Additional_lang,
            person.Home_lang,
            person.Degree,
            person.Science,
            person.Birthmonth

        ]
    ]

    cols = [
        "Female",
        "Tenure",
        "Birthyear",
        'Status',
        'Geography',
        'Province',
        'Matric',
        'Schoolquintile',
        'Diploma',
        'Math',
        'Mathlit',
        'Additional_lang',
        'Home_lang',
        'Degree',
        'Science',
        'Birthmonth'
    ]

    datafr = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=datafr)
    result = (
        {"Most probably to be employed in the year"}
        if predictions == [1]
        else {"Most probably to be unemployed in the year"}
        if predictions == [0]
        else  "Error result"
    )
    return result
    
@app.post("/clustering")
def clustering():
    """
    Кластеризация данных из файла
    """
    clust = pipeline_clustering(config_path=CONFIG_PATH)
    assert isinstance(clust, pd.DataFrame), "Результат не соответствует типу DataFrame"
    return "Successfully 5 clusters"


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
