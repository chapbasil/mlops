"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import os
import joblib
import json
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.cluster import  SpectralClustering
from sklearn.preprocessing import Normalizer
from src.data.get_data import get_dataset
from .transform import *


def pipeline_clustering(config_path: str, **kwargs) -> pd.DataFrame:
    """
    Получения данных и кластеризация
    :param config_path: путь до файла с конфигурациями
    :return: None
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]
    
    # get data
    get_data = get_dataset(dataset_path=preprocessing_config["train_path"])
    
    # replace values
    data = replace_values(data=get_data, map_change_columns=preprocessing_config["map_change_columns"])
    data = data.drop(preprocessing_config["drop_columns_clust"], axis=1)

    # заполним пропуски нулями и модой 
    data['Tenure'] = np.where(data.Status == 'studying', 0, data.Tenure)
    data['Tenure'] = np.where(data.Status == 'other', 0, data.Tenure)
    data['Tenure'] = data.Tenure.fillna(data.Tenure.mode()[0])
    data = data.fillna('None')
    
    # создадим бины, для каждого признака свои значения
    data['Birthyear'] = data["Birthyear"].apply(lambda x: get_bins_(x))       
    data['Tenure'] = data["Tenure"].apply(lambda x: get_bins_t(x))
    
    data.Female = data.Female.astype('object')
    data = data.drop(columns=train_config["target_column"], axis=1)
    
    # binary
    train_data = pd.get_dummies(data)
    
    for i in list(train_data):
        if '_None'  in i:
            train_data = train_data.drop(i, axis=1)
    #normalizing
    tlogbin =  Normalizer().fit(train_data)
    normtlogbin = tlogbin.transform(train_data)
    pca = PCA(n_components=train_config["n_components"], random_state=train_config["random_state"])

    X_embedding = pca.fit_transform(normtlogbin)
    
    # clustering
    clf_final = SpectralClustering(n_clusters=train_config["n_clusters"], \
                                   affinity=train_config["affinity"], random_state=train_config["random_state"])
    clf_final.fit(train_data)
    
    # save result (study, model)
    joblib.dump(clf_final, os.path.join(train_config["clust_path"]))
       
    df_cleaned = train_data.assign(clusters=clf_final.fit_predict(train_data))
    
     # переименуем колонки для наглядности
    df_cleaned = df_cleaned.rename(columns = (preprocessing_config["name_change_columns"]))
    df_cleaned.to_csv(preprocessing_config["clust_proc"], index=False)
    
    return df_cleaned
