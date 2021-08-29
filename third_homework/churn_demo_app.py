import streamlit as st
import pandas as pd
import numpy as np
import joblib
from churn_classifier import ChurnClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# import re
# from PIL import Image


df = joblib.load('dataset.pkl')

st.title('Демонстрационный стенд по прогнозу оттока клиентов телеком оператора')

st.markdown('Метрики качества текущей модели:')
col1, col2, col3 = st.columns(3)
col1.metric("Roc auc", "0.928")
col2.metric("Precision", "0.893")
col3.metric("Recall", "0.676")

st.markdown('''Сервис позволяет оценить работу построенной модели. 
            На основе входных данных алгоритм выдаёт аналитику по оттоку в выбранных штатах.''')

with st.expander("Посмотреть инструкцию"):
    st.write("""
        1) Вставьте ссылку в окно ниже\n
        (тестовую ссылку можно взять во вкладке "Открыть ссылку на датасет")
        2) Выбрать метрику\n
        3) Выбрать штаты\n
        4) Нажать кнопку "Сделать предсказание"
                """)

with st.expander("Открыть ссылку на датасет"):
    st.write('https://drive.google.com/uc?id=1yUAm-9nQ-pmvYAVH6uvR35KKNrc7mECO')


st.markdown('\nВводимый датасет должен состоять из следующих полей:')
df = df.drop(['Churn', 'Total day minutes', 'Total eve minutes', 'Total night minutes',
              'Total intl minutes', 'Total charge'], axis=1)
df[df.index < 3]

with st.form('\nform'):
    dataset_URL = st.text_area('Вставьте ссылку в окно ниже: ', 'вставьте ссылку сюда...')
    option = st.selectbox('Выберите метрику',
    ('Клиенты, склонные к оттоку(в %)', 'Клиенты, склонные к оттоку(количество)',
     'Лояльные клиенты(в %)', 'Лояльные клиенты(количество)'))
    selected_states = st.multiselect("Выберите штат(-ы)", df['State'].unique())
    submit_button = st.form_submit_button('Сделать предсказание')




if submit_button:
    # if re.findall(r"https://drive\Wgoogle\Wcom/\S{39}", dataset_URL) == []:


    if not selected_states:
      st.warning('Пожалуйста введите интересующий(-ие) штат(-ы).')
      st.stop()

    df_to_predict = None
    while df_to_predict is None:
        try:
            df_to_predict = pd.read_csv(dataset_URL)
        except Exception as e:
            st.warning('Пожалуйста введите ссылку на датасет в корректном формате.')
            st.stop()

    if len(df.columns) == len(df_to_predict.columns):
        if set(df.columns) != set(df_to_predict.columns):
            st.warning('Пожалуйста введите ссылку на корректный датасет.')
            st.stop()
    else:
        if len(set(df.columns) - set(df_to_predict.columns)) > 0:
            st.write(f"В загруженном датасете не хватает признаков: {set(df.columns) - set(df_to_predict.columns)}")
        if len(set(df_to_predict.columns) - set(df.columns)) > 0:
            st.write(
                f"В загруженном датасете присутствуют лишние признаки: {set(df_to_predict.columns) - set(df.columns)}")
        st.warning('Пожалуйста введите ссылку на корректный датасет.')
        st.stop()

    model = ChurnClassifier()
    predict = model.predict_customer_churn(df_to_predict)
    st.markdown(f"<h3 style='text-align: center; color: black;'>Статистика по всему набору данных </h3>"
                f"<p style='font-size':11pt><br> Клиенты, склонные к оттоку: {round(len(predict[predict.Churn==True]) / len(predict)*100)}%, "
                f"{predict[predict.Churn==True]['Churn'].sum()} человек"
                f"<br> Лояльные клиенты: {round((1 - predict[predict.Churn==True]['Churn'].sum() / len(predict)) * 100)}%, "
                f" {len(predict) - predict[predict.Churn==True]['Churn'].sum()} человек"
                f"<br>Всего клиентов: {len(predict)}"
                f"</p>"
                f"<h3 style='text-align: center; color: black;'>Статистика по выбранным штатам </h3>",
                unsafe_allow_html=True)
    encoder = joblib.load('enc.pkl')
    data = []
    height = 8
    width = 10
    selected_states = encoder.classes_
    if option == 'Клиенты, склонные к оттоку(в %)':
        for i in selected_states:
            data.append(100 * predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum() / len(predict[predict['State'] == encoder.transform([i])[0]]))
        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)

    elif option == 'Клиенты, склонные к оттоку(количество)':
        for i in selected_states:
            data.append(predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum())
        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)

    elif option == 'Лояльные клиенты(в %)':
        for i in selected_states:
            data.append(1 - predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum() / len(predict[predict['State'] == encoder.transform([i])[0]]))
        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)

    elif option == 'Лояльные клиенты(количество)':
        for i in selected_states:
            data.append(len(predict[predict['State'] == encoder.transform([i])[0]]) - predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum())
        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)
