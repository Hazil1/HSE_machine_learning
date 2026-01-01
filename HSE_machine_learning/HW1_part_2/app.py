import pickle
import re

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def split_torque(torque):
    '''
    Функция делит столбец torque на два столбца (torque, max_torque_rpm), преобразует
    крутящий момент из 'kgm' в 'Nm' и избавляется от единиц измерения в столбце torque
    '''
    if not pd.isna(torque):
        try:
            if '@' in torque:
                parts = torque.split('@', 1)
                torque = parts[0].strip()
                max_torque_rpm = parts[1].strip()
            elif 'at' in torque:
                parts = torque.split('at', 1)
                torque = parts[0].strip()
                max_torque_rpm = parts[1].strip()

            # Преобразуем `kgm` в `Nm` и избавимся от единиц измерения
            if 'kgm' in torque.lower() or 'kgm' in max_torque_rpm.lower():
                torque = float(re.sub(r'[^\d.]', '', torque)) * 9.8
            elif 'nm' in torque.lower():
                torque = float(re.sub(r'[^\d.]', '', torque))
            else:
                torque = float(torque)
            return torque
        except:
            return np.nan
    else:
        return np.nan


st.title("Оценка стоимости автомобиля с помощью линейной регрессии")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Загруженные данные (данные без преобразования)")
    st.dataframe(df)

    # Выполним преобразование признаков
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].str.replace(r'[^0-9.]', '', regex=True).astype(float)

    df['max_power'] = df['max_power'].replace('', np.nan)
    df['torque'] = df['torque'].apply(split_torque).apply(pd.Series)

    # Вывод графиков по числовым признакам
    st.subheader("Распределение числовых признаков")

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Выберите признак для вывода гистограммы", numeric_columns)

    fig = px.histogram(
        df, x=selected_col,
        marginal="box",
        title=f"Распределение признака `{selected_col}`",
        color_discrete_sequence=['#a4a5b9']
    )

    st.plotly_chart(fig, use_container_width=True)

    # Вывод информации по категориальным признакам
    st.subheader("Распределение числовых признаков")
    # Марка автомобиля
    df['marque'] = df['name'].str.lower().str.split(' ').str[0]

    st.subheader("Распределение марок автомобилей")
    fig = px.histogram(
        df,
        x="marque",
        color="marque",
        text_auto=True,
        title="Количество автомобилей по автопроизводителям",
        width=1500,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Отобразим остальные категориальные признаки
    cat_features = ['fuel', 'seller_type', 'transmission', 'owner']
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for i, feature in enumerate(cat_features):
        fig = px.histogram(
            df,
            x=feature,
            color=feature,
            text_auto=True,
            title=f"Распределение: {feature}",
            height=350
        )
        fig.update_layout(showlegend=False)
        cols[i].plotly_chart(fig, use_container_width=True)

    num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power',	'torque']
    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'marque', 'seats']
    price = df['selling_price']
    df = df[num_features + cat_features]


    # Вывод оценок по данным
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(df)
    delta = price - predictions

    df['Predictions'] = pd.Series(predictions)
    st.dataframe(df)

    # Вывод распределения оценок
    fig = px.histogram(
        df, x='Predictions',
        marginal="box",
        title=f"Распределение оценок стоимости автомобиля по модели",
        color_discrete_sequence=['#a4a5b9']
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Загрузите файл в формате .csv")
