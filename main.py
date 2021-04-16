import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt


def run(companyIndex, date1, date2, date3, predictDays, numberOfEpochs, saveBool,loadModelBool):
    tf.compat.v1.disable_eager_execution()
    # Importing data
    company = companyIndex
    predDays = predictDays
    start = date1
    end = date2
    endTest = date3
    data = web.DataReader(company, 'yahoo', start, end)
    data = data.drop(['Open', 'High', 'Volume', 'Adj Close', 'Low'], 1)
    data.drop(data.columns[0], axis=1)

    data_test = web.DataReader(company, 'yahoo', end, endTest)
    data_test = data_test.drop(['Open', 'High', 'Volume', 'Adj Close', 'Low'], 1)
    data_test.drop(data_test.columns[0], axis=1)

    # Preparing data
    data = data.values
    startInd = 0
    endInd = data.shape[0]
    data_train = data[np.arange(startInd, endInd), :]  # przedział [start,2,3,4,...,end]

    data_test = data_test.values
    data_test = data_test[np.arange(startInd, endInd), :]

    # Scalling data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    scaler.fit(data_test)
    data_test = scaler.transform(data_test)

    # makking arrays from data
    rws = data_train.shape[0] - predDays
    rws2 = data_test.shape[0] - predDays

    k = data_train.shape[0]
    Y_train = data_train[predDays:k, 0]
    data_train = data_train.T

    k = data_test.shape[0]
    Y_test = data_test[predDays:k, 0]
    data_test = data_test.T

    j = 0
    X_train = np.zeros((rws, predDays))
    for i in range(predDays, data_train.shape[1]):
        X_train[j, :] = data_train[0, i - predDays:i]
        j = j + 1

    j = 0
    X_test = np.zeros((rws2, predDays))
    for i in range(predDays, data_test.shape[1]):
        X_test[j, :] = data_test[0, i - predDays:i]
        j = j + 1

    # Number of neurons
    neuronNum1 = 256
    neuronNum2 = 128
    neuronNum3 = 64
    neuronNum4 = 32
    trainNum = X_train.shape[1]

    # Starting session
    session = tf.compat.v1.InteractiveSession()

    # Making placeolders for X and Y
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, trainNum])
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

    
    # Making Initializer
    # relu: ~1.43, tanh: ~1.15.
    initializerr = tf.compat.v1.uniform_unit_scaling_initializer(factor=1.43)
    biasInitializerr = tf.zeros_initializer()

    # Hidden layers and weights
    wLayer1 = tf.Variable(initializerr([trainNum, neuronNum1]))
    biasL1 = tf.Variable(biasInitializerr([neuronNum1]))
    wLayer2 = tf.Variable(initializerr([neuronNum1, neuronNum2]))
    biasL2 = tf.Variable(biasInitializerr([neuronNum2]))
    WLayer3 = tf.Variable(initializerr([neuronNum2, neuronNum3]))
    biasL3 = tf.Variable(biasInitializerr([neuronNum3]))
    WLayer4 = tf.Variable(initializerr([neuronNum3, neuronNum4]))
    biasL4 = tf.Variable(biasInitializerr([neuronNum4]))

    # Output weights
    W_out = tf.Variable(initializerr([neuronNum4, 1]))
    bias_out = tf.Variable(biasInitializerr([1]))

    # Calculating Hidden Layers
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, wLayer1), biasL1))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, wLayer2), biasL2))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2 , WLayer3), biasL3))
    layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, WLayer4), biasL4))

    # Model parameters
    # Calculating Output Layer
    outputLayer = tf.transpose(tf.add(tf.matmul(layer4, W_out), bias_out))
    # MSE - Cost Function
    mse = tf.reduce_mean(tf.math.squared_difference(outputLayer, Y))
    # Adam Optimazer
    opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)
    # Running session
    session.run(tf.compat.v1.global_variables_initializer())

    # Size of batch ~ 9% numer of rows
    batchSize = int(np.floor(0.09 * trainNum))
    mse_train = []
    mse_test = []
    # Prepare for plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(Y_test)
    line2, = ax1.plot(Y_test * 0.5)

    #Prepare Saver
    saver = tf.compat.v1.train.Saver()
    if loadModelBool:
        saver.restore(session, "zapismodelu.ckpt")
    # Train and test
    epochs = numberOfEpochs
    for e in range(epochs):

        # Select Training data
        randIndex = np.random.permutation(np.arange(len(Y_train)))
        X_train = X_train[randIndex]
        Y_train = Y_train[randIndex]

        # Minibatch 
        for i in range(0, len(Y_train) // batchSize):
            start = i * batchSize
            batchX = X_train[start:start + batchSize]
            batchY = Y_train[start:start + batchSize]
            # Optimazer
            session.run(opt, feed_dict={X: batchX, Y: batchY})
            # Show progress
            if np.mod(i, 5) == 0:
                # MSE train and test
                mse_train.append(session.run(mse, feed_dict={X: X_train, Y: Y_train}))
                mse_test.append(session.run(mse, feed_dict={X: X_test, Y: Y_test}))
                print('MSE For Training Data: ', mse_train[-1])
                print('MSE For Testing Data: ', mse_test[-1])
                # Prediction
                pred = session.run(outputLayer, feed_dict={X: X_test})
                pred2 = session.run(outputLayer, feed_dict={X: X_train})
            if e == epochs-1:
                line1.set_ydata(Y_test)
                line2.set_ydata(pred)
                plt.title('Test data')
                plt.savefig('img/final.jpg')
                line1.set_ydata(Y_train)
                line2.set_ydata(pred2)
                plt.title('Train data')
                plt.savefig('img/finalTrain.jpg')


    mse_final = session.run(mse, feed_dict={X: X_test, Y: Y_test})
    print(mse_final)

    batchX = X_test[0:1]
    prediction = session.run(outputLayer, feed_dict={X: batchX})
    prediction = scaler.inverse_transform(prediction)
    print("Przewidziana cena: ", prediction)
    real = Y_train[61]
    real = real.reshape(-1, 1)
    real = scaler.inverse_transform(real)
    print("Rzeczywista cena:", real)
    if (saveBool):
        saver.save(session, "zapismodelu.ckpt")


#run(companyIndex='FB', predictDays=60, numberOfEpochs=5, saveBool=True, date1=0, date2=0, date3=0)

