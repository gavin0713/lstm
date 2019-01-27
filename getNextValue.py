from keras.models import load_model
from lstm_model.lstm import gj_series_to_supervised
from pandas import read_table
from sklearn.preprocessing import MinMaxScaler

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
    reframed = gj_series_to_supervised(scaled, cols-1, 1, False, True)
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


# a实测
#     def nash(a, b):
#         a_mean = np.mean(a)
#         b_mean = np.mean(b)
#         c = np.zeros(len(a))
#         d = np.zeros(len(a))
#         for i in range(len(d) - 1):
#             c[i] = (a[i] - b[i]) ** 2
#             d[i] = (a[i] - a_mean) ** 2
#         c_sum = np.sum(c)
#         d_sum = np.sum(d)
#         return 1 - c_sum / d_sum


if __name__ == '__main__':


    model = load_model('model.h5')

    target, values, scaler, valmax, valmin = get_data('hydrodata')
    n_train_hours = 365 * 2
    train, test = get_data_simple(values, n_train_hours)

    # split into input and outputs
    train_X, train_Y = train[:, :-1], train[:, -1]
    test_X, test_Y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)



    predict = model.predict(train_X)


    # .reshape(len(myx))

    print("train_X:",len(train_X))
    print("predict:", len(predict))

    realpredict = get_real_values(predict, valmax, valmin)


    fout = open('out', 'w')


    for i in range(len(myy)):
        fout.write(str(realpredict[i]) + "," + str(target[i]) + '\n')
    fout.close()




    print(len(myx))
    print(len(myy))
    print(len(predict))

    print("nash:" + str(nash(myy[:-1], predict[1:])))
    #
    print("nash:" + str(nash(myy, predict)))


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

