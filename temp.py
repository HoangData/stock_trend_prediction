import datetime
import os
import plotly.graph_objects as go
import keras.models
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import yfinance as yf



def progress(data, start, end):
    st.header("Data from "+ str(start) + " to "+ str(end))
    st.dataframe(data.describe())


    st.header("Close price vs Time chart")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data.Close)
    st.pyplot(fig)

    st.header("Close price vs Time chart with 100MA")
    ma100 = data.Open.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(data.Close)
    st.pyplot(fig)

    st.header("Close price vs Time chart with 100MA & 200MA")
    ma100 = data.Open.rolling(100).mean()
    ma200 = data.Open.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(data.Close)
    st.pyplot(fig)



    #finance chart
    # fig = plt.figure(figsize=(12,6))
    # fig = go.Figure(data=[go.Candlestick(x=df['DATE'],
    #                 open=df['OPEN'],
    #                 high=df['HIGH'],
    #                 low=df['LOW'],
    #                 close=df['CLOSE'])])
    #
    # st.plotly_chart(fig, use_container_width=True)

    # data_train = data[data['DATE'] < '2019-01-02'].copy()
    # data_test = data[data['DATE'] >= '2019-01-02'].copy()
    # data_training = data_train.copy()


    past_60_days = yf.download(str(input),
                      start = str(start - datetime.timedelta(days=100)),
                      end = str(start),
                      progress=False)
    if len(past_60_days) >= 60:
        past_60_days = past_60_days.tail(60)

    data_test = past_60_days.append(data, ignore_index=True)

    # Dropping 'DATE'
    # data_train = data_train.drop(['DATE'], axis=1)
    data_test = data_test.drop(['Adj Close'], axis=1)

    #
    sc = MinMaxScaler()
    # data_train = sc.fit_transform(data_train)
    data_test = sc.fit_transform(data_test)
    #
    # X_train = []
    # y_train = []

    # for i in range(60, data_train.shape[0]):
    #     X_train.append(data_train[i - 60:i])
    #     y_train.append(data_train[i, 0])

    # X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []

    for i in range(60, data_test.shape[0]):
        X_test.append(data_test[i - 60:i])
        y_test.append(data_test[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)


    model = keras.models.load_model("mymodel")

    y_pred = model.predict(X_test)

    scale = 1 / 1.26726651e-03
    y_pred = y_pred * scale
    y_test = y_test * scale

    st.header("Prediction & Original")
    fig =plt.figure(figsize=(16,6))
    plt.plot(y_test, color = 'LimeGreen', label = 'Real VNIndex Stock Price')
    plt.plot(y_pred, color = 'Gold', label = 'Predicted VNIndex Stock Price, LSTM')
    plt.title('VNIndex Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('VNIndex Stock Price')
    plt.legend()
    st.pyplot(fig)


if __name__ == "__main__":

    file = open("whitelist.txt", 'r')
    listTicker = file.readlines()
    slb = st.sidebar.radio("WHITE LIST", listTicker)

    st.title("Stock Trend Prediction Using RNN Algorithm")
    link = "Stock Ticker [https://finance.yahoo.com/trending-tickers](https://finance.yahoo.com/trending-tickers)"
    st.markdown(link, unsafe_allow_html=True)
    input = st.text_input("Enter ticker:", '')
    btn = st.button("Add white list")


    col1, col2 = st.columns(2)

    with col1:
        start = st.date_input("Start date")
    with col2:
        end = st.date_input("End date")

    if input != "":
        data = yf.download(str(input),
                           start=str(start),
                           end=str(end),
                           progress=False)
        progress(data, start, end)


