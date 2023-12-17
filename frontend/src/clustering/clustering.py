"""
Программа: Кластеризация
Версия: 1.0
"""

import os
import json
import requests
import yaml
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data.get_data import get_dataset


def clustering_(input_path: str, config: dict, endpoint: object) -> matplotlib.figure.Figure:
    """
     Кластеризация c выводом описаний кластеров
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    df_cleaned = get_dataset(dataset_path = input_path)
    
    # Для вывода барплотом необходимо преобразовать выгруженный с бэкенда датасет
    # Датасет в виде, сохраненном на бэкенде, будет необходим для других функций в проекте, 
    # поэтому преобразуем данные из него непосредственно перед выводом графиков
    # Считаем кол-во объектов в каждом кластере
    count = df_cleaned.groupby('clusters').count().iloc[:, :1].T.values

    # Нормируем кол-во объектов в каждом из признаков в кластере на общее кол-во объектов в кластере
    normalize = df_cleaned.groupby('clusters').sum().T / count * 100
    
    fig, axes = plt.subplots(nrows=5, figsize=(15, 50))
    sns.barplot(data=normalize.iloc[:,0:1].sort_values(by=0, ascending=False).T.iloc[:, 0:15], orient='h', ax=axes[0]).set(title="1 кластер")
    st.markdown("""1 кластер:
    Городские студентки всех возрастов без стажа работы. У некоторых уровень владения базовым английским выше среднего.""")
    st.markdown("""2 кластер:
Девушки до 23 лет, в основном городские, со школьным аттестатом без высшего образования, которые в основном ни работают, ни учатся. У некоторых уровень владения английским выше среднего.""")
    st.markdown("""3 кластер:
    Женщины старше 23 лет, в основном городские, со школьным аттестатом без высшего образования, которые в основном ни работают, ни учатся. Немногие рабоают по найму.""")
    st.markdown("""4 кластер:
    Городские мужчины всех возратов со школьным аттестатом без высшего образования, в основном безработные. Немногие рабоают по найму. Большая часть молодые.""")
    st.markdown("""5 кластер:
    Мужчины старше 23 лет, со школьным аттестатом без вышего образования, занятые учебой или прочей деятельностью.
В целом, "Не безработные мужчины".""")
    sns.barplot(data=normalize.iloc[:,1:2].sort_values(by=1, ascending=False).T.iloc[:, 0:15], orient='h', ax=axes[1]).set(title="2 кластер")
    sns.barplot(data=normalize.iloc[:, 2:3].sort_values(by=2, ascending=False).T.iloc[:, 0:15], orient='h', ax=axes[2]).set(title="3 кластер")
    sns.barplot(data=normalize.iloc[:, 3:4].sort_values(by=3, ascending=False).T.iloc[:, 0:16], orient='h', ax=axes[3]).set(title="4 кластер")
    sns.barplot(data=normalize.iloc[:, 4:5].sort_values(by=4, ascending=False).T.iloc[:, 0:15], orient='h', ax=axes[4]).set(title="5 кластер")
    return fig
    
def clustering_set(input_path: str, config: dict, endpoint: object) -> pd.DataFrame:
    """
     Датасет для прогноза по кластерам
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """

    df_cleaned = get_dataset(dataset_path = input_path)
    df_targ = get_dataset(dataset_path = config["preprocessing"]["train_path"])
    Pred = pd.concat([df_cleaned[['clusters']], df_targ[config["preprocessing"]["target_column"]].to_frame()], axis=1)
    Pred = Pred.replace({'clusters' : { 0 : 'Studying women', 1 : 'Unemployed girls', 2: 'Unemployed elder women', \
                          3: 'Unemployed men', 4: 'Employed or studying men'}})
    return Pred
