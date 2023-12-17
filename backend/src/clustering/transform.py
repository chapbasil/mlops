"""
Программа: Предобработка данных для кластеризации
Версия: 1.0 Pymagic
"""

import json
import warnings
import pandas as pd
import numpy as np
import yaml

config_path= "../config/params.yml"

warnings.filterwarnings("ignore")


def replace_values(data: pd.DataFrame, map_change_columns: dict) -> pd.DataFrame:
    """
    Замена значений в датасете
    :param data: датасет
    :param map_change_columns: словарь с признаками и значениями
    :return: датасет
    """
    return data.replace(map_change_columns)



def binar(data: pd.DataFrame):
    """
    Функция бинаризации, при необходимости
    data: датасет
    """
    data = pd.get_dummies(data)
    return data
    
def get_bins_(data: int) -> str:
    """
    Генерация бинов для признака "Год рождения"
    """
    if isinstance(data, (int, float)):
        if data <= 1999:
            return 'before_2000'
        elif data >= 2000:
            return '2000х'


def get_bins_t(data: int) -> str:
    """
    Генерация бинов для признака "Длительность статуса"
    """
    if isinstance(data, (int, float)):
        if data == 0.0:
            return 'NoTenure'
        elif data != 0.0:
            return 'AnyTenure'
