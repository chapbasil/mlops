"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(input_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(input_path) as file:
        input_df = json.load(file)

    female = st.sidebar.selectbox("Female", (input_df["Female"]))
    tenure = st.sidebar.slider(
        "Tenure", min_value=0.0, max_value=10000.0)
    birthyear = st.sidebar.slider(
        "Birtrhyear", min_value=min(input_df["Birthyear"]), max_value=max(input_df["Birthyear"])
    )
    status = st.sidebar.selectbox("Status", (input_df["Status"]))
    geography = st.sidebar.selectbox("Geography", (input_df["Geography"]))
    province = st.sidebar.selectbox("Province", (input_df["Province"]))
    matric = st.sidebar.selectbox("Matric", (input_df["Matric"]))
    schoolquintile = st.sidebar.selectbox("Schoolquintile", (input_df["Schoolquintile"]))
    diploma = st.sidebar.selectbox("Diploma", (input_df["Diploma"]))
    math = st.sidebar.selectbox("Math", (input_df["Math"]))
    mathlit = st.sidebar.selectbox("Mathlit", (input_df["Mathlit"]))
    additional_lang = st.sidebar.selectbox("Additional_lang", (input_df["Additional_lang"]))
    home_lang = st.sidebar.selectbox("Home_lang", (input_df["Home_lang"]))
    degree = st.sidebar.selectbox("Degree", (input_df["Degree"]))
    science = st.sidebar.selectbox("Science", (input_df["Science"]))
    birthmonth = st.sidebar.selectbox(
        "Birthmonth", (sorted(input_df["Birthmonth"]))
    )
    

    dict_data = {
        "Female": female,
        "Tenure": tenure,
        "Birthyear": birthyear,
        "Status": status,
        "Geography": geography,
        "Province": province,
        "Matric": matric,
        "Schoolquintile": schoolquintile,
        "Diploma": diploma,
        "Math": math,
        "Mathlit": mathlit,
        "Additional_lang": additional_lang,
        "Home_lang": home_lang,
        "Degree": degree,
        "Science": science,
        "Birthmonth": birthmonth
        
    }

    st.write(
        f"""### Данные человека:\n
    1) Female: {dict_data['Female']}
    2) Tenure: {dict_data['Tenure']}
    3) Status: {dict_data['Status']}
    4) Geography: {dict_data['Geography']}
    5) Province: {dict_data['Province']}
    6) Matric: {dict_data['Matric']}
    7) Schoolquintile: {dict_data['Schoolquintile']}
    8) Diploma: {dict_data['Diploma']}
    9) Math: {dict_data['Math']}
    10) Mathlit: {dict_data['Mathlit']}
    11) Additional_lang: {dict_data['Additional_lang']}
    12) Home_lang: {dict_data['Home_lang']}
    13) Degree: {dict_data['Degree']}
    14) Science: {dict_data['Science']}
    15) Birthyear: {dict_data['Birthyear']}
    16) Birthmonth: {dict_data['Birthmonth']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output[0]}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:100]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_)
