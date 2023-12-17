"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from src.data.get_data import load_data, get_dataset
from src.plotting.charts import barplot_group
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file
from src.clustering.clustering import clustering_, clustering_set

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.markdown("# Описание проекта")
    st.title("MLOps project:  Will the person find a job or not in the year?")
    st.image(
        "https://img4.teletype.in/files/f4/5a/f45a687e-0d8b-427b-828e-8262ece50e19.jpeg",
        width=600,
    )

    st.write(
        """
        Целью этой задачи является создание модели машинного обучения, которая прогнозирует занятость молодежи на основе данных исследований рынка труда в Южной Африке.

        Это решение поможет таким организациям, как Predictive Insights, получить базовый прогноз результатов трудоустройства молодых людей, что позволит им разрабатывать и тестировать меры, которые помогут молодежи выйти на рынок труда или улучшить свои заработки."""
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
- Status - Тип занятости
- Tenure - Длительность статуса в днях. Если человек работает или учится, это то стаж работы или учебы, а если безработный, то, как давно человек без работы. Например, чем больше срок, тем лучше показатель для работающих и хуже для безработных.
- Geography - Годод/село
- Province - Регион ЮАР     
- Matric - Наличие аттестата об окончании школы ЮАР: 1 - да, 0 - нет
- Degree  - Наличие научной степени 
- Diploma - Наличие диплома о высшем образовании
- Schoolquintile - Квантиль школы в по систему ЮАР (см. примечание ниже)
- Math и Mathlit - два варианта математики, взятые в матрице. Ученики должны сдавать один тип и не могут оба
- Additional_lang - Школный бал в процентах по начальному англискому языку
-- Обратим внимание, что английский - официальный язык ЮАР, но по факту в рейтинге языков, на которых говорят граждане, он занимает пятое место. Это не самый распространённый язык для повседневного общения 
- Home_lang - Школный бал в процентах по угулбленному англискому языку
- Science - Школный бал в процентах по предмету "Наука" (В ЮАР в один предмет входят физика и химия)
- Female - Женский пол: 1 - да, 0 - мужской
- Sa_citizen  - Наличие гражданства ЮАР: 1 - да, 0- нет
- Birthyear - Год рождения
- Birthmonth - Месяц рождения  
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["train_path"])
    st.write(data.head())

    # plotting with checkbox
    edu_ = st.sidebar.checkbox("Ожидание трудоустройства для людей со школьным аттестатом")
    diploma_ = st.sidebar.checkbox("Ожидание трудойстройства для людей с дипломом о высшем образовании")
    degree_ = st.sidebar.checkbox(
        "Ожидание трудоустройства для людей c ученой степенью"
    )
    urbanisation_ = st.sidebar.checkbox("Ожидание трудоустройства в зависимости от урбанизации")
    gender_ = st.sidebar.checkbox("Ожидание трудоустройства в зависимости от пола")
    status_ = st.sidebar.checkbox("Ожидание трудоустройства в зависимости от статуса")
    school_ = st.sidebar.checkbox("Ожидание трудоустройства по уровням школьного образования")
    province_ = st.sidebar.checkbox("Ожидание трудоустройства по провинциям")
    
    if edu_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Matric",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудоустройства для людей со школьным аттестатом",
            )
        )
    if diploma_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Diploma",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудойстройства для людей с дипломом о высшем образовании",
            )
        )
    if degree_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Degree",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудойстройства для людей с дипломом о высшем образовании",
            )
        )
    if urbanisation_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Geography",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудойстройства в зависимости от урбанизации",
            )
        )

    if gender_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Female",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудойстройства в зависимости от пола",
            )
        )
        
    if status_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Status",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудойстройства в зависимости от статуса",
            )
        )
        
    if school_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Schoolquintile",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудойстройства по уровням школьного образования",
            )
        )
    
    if province_:
        st.pyplot(
            barplot_group(
                data=data,
                col_main="Province",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудоустройства по провинциям",
            )
        )
def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    input_path = config["preprocessing"]["input_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(input_path=input_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")

def clustering():
    """
    Кластеризация с описание кластеров
    """
    
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        endpoint = config["endpoints"]["clustering"]
    
    if st.button("Start clustering"):
        fig = clustering_(input_path = config["preprocessing"]["clust_proc"], config = config, endpoint = endpoint)
        st.pyplot(fig)
       
        st.pyplot(
            barplot_group(
                data=clustering_set(input_path = config["preprocessing"]["clust_proc"], config = config, endpoint = endpoint),
                col_main="clusters",
                col_group=config["preprocessing"]["target_column"],
                title="Ожидание трудоустройства для разных кластеров"
            )
           )
        
def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
        "Clustering": clustering
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()
        

if __name__ == "__main__":
    main()
