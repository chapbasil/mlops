"""
Программа: Получение данных из файла
Версия: 1.0 Pymagic
"""

from typing import Text
import pandas as pd


def get_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_csv(dataset_path)
