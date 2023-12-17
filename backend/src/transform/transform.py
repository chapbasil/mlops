"""
Программа: Предобработка данных
Версия: 1.0 Pymagic
Закомментированы функции, которые могли быть полезны для разных экспериментов предобработки данных
Файл рабочий, просьба не удалять =)
"""

import json
import warnings
import pandas as pd
import numpy as np
import yaml

config_path= "../config/params.yml"
from ..data.split_dataset import get_train_test_data

warnings.filterwarnings("ignore")


def replace_values(data: pd.DataFrame, map_change_columns: dict) -> pd.DataFrame:
    """
    Замена значений в датасете
    :param data: датасет
    :param map_change_columns: словарь с признаками и значениями
    :return: датасет
    """

    return data.replace(map_change_columns)


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")

def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str, **kwargs) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)
    
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    train_config = config["train"]
        
    column_sequence = unique_values.keys()
    #  assert set(column_sequence) == set(data.columns), column_sequence
    if set(column_sequence) != set(data.columns):
        for j in list(column_sequence):
            if j not in list(data):
                data[j] = 0 
              
    return data[column_sequence] 


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для сравнения вводимых данных
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)

def save_input_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, input_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    input_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для ввода в UI
    # для того, чтобы типы данных не смешивались, во избежание ошибки, столбцы с числовыми значениями заполним нулями
    input_df[["Degree", "Diploma", "Schoolquintile", "Matric"]] = input_df[["Degree", \
              "Diploma", "Schoolquintile", "Matric"]].fillna(0.0)
    input_df = input_df.fillna("None")
    
    dict_input = {key: input_df[key].unique().tolist() for key in input_df.columns}
    with open(input_path, "w") as file:
        json.dump(dict_input, file)

def binar(data: pd.DataFrame):
    """
    Функция бинаризации, при необходимости
    data: датасет
    """
    data = pd.get_dummies(data)
    return data
    
    
def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = False, **kwargs):
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :param flg_evaluate: флаг для evaluate
    :return: датасет
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    train_config = config["train"]
    
    # значения для добавления признаков при бинаризации и если поданы не все признаки
    unique_values_path=kwargs["unique_values_path"]
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)
        
    # значения для ввода в UI
    input_path=kwargs["input_path"]
    with open(input_path) as json_file:
        input_values = json.load(json_file)
    
    try: # исключение ошибки ручного ввода, в который подются только необходимые признаки
       data = data.drop(kwargs["drop_columns"], axis=1)
    except:
        pass

    if flg_evaluate:
        pass
    else:
        save_input_data(
            data=data,
            drop_columns=kwargs["drop_columns"],
            target_column=kwargs["target_column"],
            input_path=kwargs["input_path"],
        )
    
    # transform values
    data = replace_values(data=data, map_change_columns=kwargs["map_change_columns"])
    
    # Часть пропусков заполним нулями
    data['Tenure'] = np.where(data['Status'] == 'studying', 0, data.Tenure)
    data['Tenure'] = np.where(data.Status == 'other', 0, data.Tenure)
    
    # Остальные в признаке заполняем модой
    data['Tenure'] = data.Tenure.fillna(data.Tenure.mode())
    
    # Логарифмируем, приводим распределение в более нормальное
    data['Birthyear'] = np.log(data['Birthyear'] + 1)
    data['Tenure'] = np.log(data['Tenure'])
    data.loc[data['Tenure'] < 0, 'Tenure'] = 0.0
    data['Tenure'] = data.Tenure.fillna(data.Tenure.mode()[0])
    
    # Заполним пропуски в остальных признаках значением 'None', т.к. они либо категориальные, либо бинарные.
    data = data.fillna('None')
    
    # Биниарзуем
    data = binar(data)
    
    # проверка dataset на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train
    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_train_data(
            data=data,
            drop_columns=kwargs["drop_columns"],
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )

   # закомменчен код для категоризации признаков, не стала удалять
   # change category types
   # dict_category = {key: "category" for key in data.select_dtypes(["object"]).columns}
   # data = transform_types(data=data, change_type_columns=dict_category)
    
    return data
