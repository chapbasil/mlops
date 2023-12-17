"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def barplot_group(
    data: pd.DataFrame, col_main: str, col_group: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика boxplot
    :param data: датасет
    :param col_main: признак для анализа по col_group
    :param col_group: признак для нормализации/группировки
    :param title: название графика
    :return: поле рисунка
    """
    data_group = (
        data.groupby([col_group])[col_main]
        .value_counts(normalize=True)
        .rename("percentage")
        .mul(100)
        .reset_index()
        .sort_values(col_group)
    )

    data_group.columns = [col_group, col_main, "percentage"]

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 7))

    ax = sns.barplot(
        x=col_main, y="percentage", hue=col_group, data=data_group, palette="ch:start=.2,rot=-.3"
    )
    for patch in ax.patches:
        percentage = "{:.1f}%".format(patch.get_height())
        ax.annotate(
            percentage,  # текст
            (
                patch.get_x() + patch.get_width() / 2.0,
                patch.get_height(),
            ),  # координата xy
            ha="center",  # центрирование
            va="center",
            xytext=(0, 10),
            textcoords="offset points",  # точка смещения относительно координаты
            fontsize=14,
        )
    plt.title(title, fontsize=20)
    plt.ylabel("Percentage", fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig

# Пока kdeplot не используем, функция для информации
def kdeplotting(
    data: pd.DataFrame, data_x: str, hue: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика kdeplot
    :param data: датасет
    :param data_x: ось OX
    :param hue: группирвока по признаку
    :param title: название графика
    :return: поле рисунка
    """
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(15, 7))

    sns.kdeplot(
        data=data, x=data_x, hue=hue, palette="rocket", common_norm=False, fill=True
    )
    plt.title(title, fontsize=20)
    plt.ylabel("Percentage", fontsize=14)
    plt.xlabel(data_x, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig
