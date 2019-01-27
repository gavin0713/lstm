from lstm_model.lstm import gj_lstm, gj_series_to_supervised, gj_lstm_ankang
from pandas import read_table
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from time import time
from numpy import concatenate
from math import sqrt
# load dataset index 径流 降雨
from keras.models import load_model

import tensorflow as tf

import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


def get_data(filepath):
    dataset = read_table(filepath, header=0,index_col=False)
    print(dataset.head())
    (rows, cols) = dataset.shape
    values = dataset.values
    target = np.copy(values[:, -1])

    valmax = np.max(target)
    valmin = np.min(target)

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    print(cols)
    reframed = gj_series_to_supervised(scaled, cols-1, 1, True, True, 30)
    # drop columns we don't want to predict
    print(reframed.head())

    # split into train and test sets
    values = reframed.values

    return target, values, scaler, valmax, valmin

def get_data_simple(values, n_train_hours):
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    return train, test


def get_real_values(values, valmax, valmin):
    k = valmax - valmin
    b = valmin

    result = np.zeros(len(values))
    for i in range(len(values)):
        result[i] = k * values[i] + b
    return result






if __name__ == '__main__':

    target, values, scaler, valmax, valmin = get_data('hydrodata')
    n_train_hours = 365*8
    train, test = get_data_simple(values, n_train_hours)
    all = np.array(values, copy=True)

    # split into input and outputs
    train_X, train_Y = train[:, :-1], train[:, -1]
    test_X, test_Y = test[:, :-1], test[:, -1]
    myx, myy = all[:, :-1], all[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    myx = myx.reshape((myx.shape[0], 1, myx.shape[1]))
    # myy = myy.reshape((myy.shape[0], 1, myy.shape[1]))
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    confit = {
        'layer': {
            'lstm': {
                'units': 30,
                'input_shape': (train_X.shape[1], train_X.shape[2])
            },
            'dense': {
                'units': 1,
                'activation': 'elu'
            }
        },
        'compile': {
            'loss': 'mse'
            , 'optimizer': 'rmsprop'
        },
        'fit': {
            'x': train_X
            , 'y': train_Y
            , 'epochs': 100
            , 'batch_size': 30
            , 'validation_data': (test_X, test_Y)
            , 'verbose': 2
            , 'shuffle': False
            , 'callbacks': [tensorboard]
        }
    }

    # model = load_model('model.h5')

    model, history = gj_lstm_ankang(confit)
    # model.save('model.h5')
    # # design network

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # validline = len(myx) - 68
    # # predictbatch = 1000
    #
    # inputdata = myx[:validline]
    #
    # outputdata = model.predict(inputdata).reshape(len(inputdata))
    # output = [outputdata[-1]]
    # for n in range(validline, len(myy)):
    #     thelast = np.copy(myx[n])
    #     print(thelast)
    #     if(n!=validline+28):
    #         thelast[0][0] = outputdata[-1]
    #     print(thelast)
    #     inputdata = np.insert(inputdata, len(inputdata), values=thelast, axis=0)
    #     outputdata = model.predict(inputdata).reshape(len(inputdata))
    #     output += [outputdata[-1]]

    # print(outputdata)




    # print(myy[len(myy) - 86])
    # print(myy[len(myy) - 85])
    # print(myy[len(myy) - 84])
    # print(myy[len(myy) - 83])
    # print(myy[len(myy) - 82])
    # print(myy[len(myy) - 81])

    # predict = model.predict(train_X, steps=2)
    predict = model.predict(myx).reshape(len(myx))

    # .reshape(len(myx))

    print("train_X:",len(train_X))
    print("predict:", len(predict))

    realpredict = get_real_values(predict, valmax, valmin)


    # output = get_real_values(output, valmax, valmin)



    # target = get_real_values(myy, valmax, valmin)

    # print('test')
    #
    # for i in range(len(output)):
    #     print(output[i])


    fout = open('out', 'w')

    print(len(realpredict))
    print(len(target))
    for i in range(len(myy)):
        fout.write(str(realpredict[i]) + "," + str(target[i]) + '\n')
    fout.close()


    # a实测
    def nash(a, b):
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        c = np.zeros(len(a))
        d = np.zeros(len(a))
        for i in range(len(d) - 1):
            c[i] = (a[i] - b[i]) ** 2
            d[i] = (a[i] - a_mean) ** 2
        c_sum = np.sum(c)
        d_sum = np.sum(d)
        return 1 - c_sum / d_sum

    print(len(myx))
    print(len(myy))
    print(len(predict))

    # print("nash:" + str(nash(myy[:-1], predict[1:])))
    # #
    print("nash:" + str(nash(myy, predict)))

    print(predict[-1])


    print("nash:" + str(nash(target, realpredict)))

    # # make a prediction
    # yhat = model.predict(test_X)
    # print(yhat)
    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    # print(inv_yhat)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:, 0]
    # # invert scaling for actual
    # test_Y = test_Y.reshape((len(test_Y), 1))
    # inv_y = concatenate((test_Y, test_X[:, 1:]), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:, 0]
    # # calculate RMSE
    # print(inv_y)
    # print(inv_yhat)
    # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % rmse)


