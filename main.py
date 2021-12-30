import csv
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


    # finance chart
    # fig = go.Figure(data=[go.Candlestick(x=data["Date"],
    #                 open=data['Open'],
    #                 high=data['High'],
    #                 low=data['Low'],
    #                 close=data['Close'])],
    #                 )
    #
    # st.plotly_chart(fig, use_container_width=True)


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


    past_60_days = yf.download(str(input),
                      start = str(start - datetime.timedelta(days=100)),
                      end = str(start),
                      progress=False)
    if len(past_60_days) >= 60:
        past_60_days = past_60_days.tail(60)

    data_test = past_60_days.append(data, ignore_index=True)

    # Dropping 'DATE'
    data_test = data_test.drop(['Date'], axis=1)
    data_test = data_test.drop(['Adj Close'], axis=1)

    #
    sc = MinMaxScaler()
    data_test = sc.fit_transform(data_test)

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
    listTicker = []

    with open("watchlist.csv", 'r') as f:
        reader = csv.reader(f, delimiter = '\n')
        for row in reader:
            listTicker.append(row[0])
            # print(row)
    slb = st.sidebar.radio("WATCHLIST", listTicker)
    st.sidebar.button("Refresh")



    st.title("Stock Trend Prediction Using RNN Algorithm")
    link = "Stock Ticker [https://finance.yahoo.com/trending-tickers](https://finance.yahoo.com/trending-tickers)"
    st.markdown(link, unsafe_allow_html=True)

    placeholder = st.empty()
    input = placeholder.text_input("Enter ticker:", value='')
    if slb in listTicker:
        input = placeholder.text_input('Enter ticker:', value=str(slb), key=1)

    col1, col2 = st.columns(2)
    with col1:
        btn_submit = st.button("Submit")
    with col2:
        btn = st.button("Add Watchlist")

    if btn:
        if input != "":
            with open("watchlist.csv", 'a', newline='') as f:
                write = csv.writer(f)
                write.writerow([input])



    col1, col2 = st.columns(2)

    with col1:
        start = st.date_input("Start date")
    with col2:
        end = st.date_input("End date")

    if btn_submit:
        if input != "":
            data = yf.download(str(input),
                               start=str(start),
                               end=str(end),
                               progress=False)
            data.reset_index(inplace=True)

            progress(data, start, end)


