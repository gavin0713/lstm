from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM,GRU


# convert series to supervised learning
# n_in + n_out, the column number of data
# data.shape = (rows, columns)

# dropnan=True, will delete the first line
# pre_target=True, will add pre target to input data
def gj_series_to_supervised(data, n_in=1, n_out=1, dropnan=True, pre_target=True, offset=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data[:, n_in:n_in + n_out])
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)

    if pre_target:
        cols.append(df.shift(offset))
        names += [('var%d(t-%d)' % (j + 1 + n_in, 1)) for j in range(n_vars - n_in)]

    df = DataFrame(data)

    # input sequence t
    cols.append(df)
    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    else:
        agg = agg.fillna(method='bfill')
    return agg

def gj_lstm_ankang(confit, debug=False):
    # design network
    if 'layer' in confit.keys() and 'compile' in confit.keys() and 'fit' in confit.keys():
        ls_layer = confit['layer']
        ls_compile = confit['compile']
        ls_fit = confit['fit']

    # model = load_model('model.h5')

    model = Sequential()

    if debug:
        print('lstm layer')
    _lstm = ls_layer['lstm']

    model.add(LSTM(units=_lstm["units"], input_shape=_lstm['input_shape']))
    model.add(Dropout(0.6))
    del ls_layer['lstm']

    for key, value in ls_layer.items():
        if (key == 'dense'):
            if debug:
                print(key, " layer")
            _dense = ls_layer['dense']
            model.add(Dense(_dense['units'], activation=_dense['activation']))
        else:
            print('没有对应层', key, value)

    if debug:
        print('compile debug')
    model.compile(loss=ls_compile['loss'], optimizer=ls_compile['optimizer'])
    if debug:
        print('fit debug')
    history = model.fit(ls_fit['x'], ls_fit['y'], epochs=ls_fit['epochs'],
                        batch_size=ls_fit['batch_size'], validation_data=ls_fit['validation_data'],
                        verbose=ls_fit['verbose'], shuffle=ls_fit['shuffle'], callbacks=ls_fit['callbacks'])

    return model, history
    # plot history


def gj_ann(confit, debug=False):
    # design network
    if 'layer' in confit.keys() and 'compile' in confit.keys() and 'fit' in confit.keys():
        ls_layer = confit['layer']
        ls_compile = confit['compile']
        ls_fit = confit['fit']

    # model = load_model('model.h5')

    model = Sequential()

    if debug:
        print('lstm layer')
    _lstm = ls_layer['lstm']

    model.add(Dense(units=_lstm["units"], input_shape=_lstm['input_shape']))
    model.add(Dropout(0.2))
    # model.add(Dense(32))
    # model.add(Dropout(0.2))
    # model.add(Dense(32))
    # model.add(Dropout(0.2))
    model.add(GRU(32, return_sequences=True))  # 返回维度为 32 的向量序列

    # model.add(Dropout(0.6))
    # model.add(GRU(32, return_sequences=True))  # 返回维度为 32 的向量序列
    model.add(Dropout(0.2))
    model.add(LSTM(32))  # 返回维度为 32 的单个向量
    # model.add(GRU(_lstm['units'], input_shape=_lstm['input_shape']))
    # model.add(Dropout(0.5))
    # model.add(LSTM(_lstm['units'], return_sequences=True, stateful=True))
    # model.add(LSTM(_lstm['units'], stateful=True))

    model.add(Dropout(0.2))
    del ls_layer['lstm']

    for key, value in ls_layer.items():
        if (key == 'dense'):
            if debug:
                print(key, " layer")
            _dense = ls_layer['dense']
            model.add(Dense(_dense['units'], activation=_dense['activation']))
        else:
            print('没有对应层', key, value)

    if debug:
        print('compile debug')
    model.compile(loss=ls_compile['loss'], optimizer=ls_compile['optimizer'])
    if debug:
        print('fit debug')
    history = model.fit(ls_fit['x'], ls_fit['y'], epochs=ls_fit['epochs'],
                        batch_size=ls_fit['batch_size'], validation_data=ls_fit['validation_data'],
                        verbose=ls_fit['verbose'], shuffle=ls_fit['shuffle'], callbacks=ls_fit['callbacks'])

    return model, history
    # plot history


def gj_lstm(confit, debug=False):
    # design network
    if 'layer' in confit.keys() and 'compile' in confit.keys() and 'fit' in confit.keys():
        ls_layer = confit['layer']
        ls_compile = confit['compile']
        ls_fit = confit['fit']

    # model = load_model('model.h5')

    model = Sequential()

    if debug:
        print('lstm layer')
    _lstm = ls_layer['lstm']

    model.add(Dense(units=_lstm["units"], input_shape=_lstm['input_shape']))
    model.add(Dropout(0.2))
    # model.add(Dense(32))
    # model.add(Dropout(0.2))
    # model.add(Dense(32))
    # model.add(Dropout(0.2))
    model.add(GRU(32, return_sequences=True))  # 返回维度为 32 的向量序列

    # model.add(Dropout(0.6))
    # model.add(GRU(32, return_sequences=True))  # 返回维度为 32 的向量序列
    model.add(Dropout(0.2))
    model.add(LSTM(32))  # 返回维度为 32 的单个向量
    # model.add(GRU(_lstm['units'], input_shape=_lstm['input_shape']))
    # model.add(Dropout(0.5))
    # model.add(LSTM(_lstm['units'], return_sequences=True, stateful=True))
    # model.add(LSTM(_lstm['units'], stateful=True))

    model.add(Dropout(0.2))
    del ls_layer['lstm']

    for key, value in ls_layer.items():
        if (key == 'dense'):
            if debug:
                print(key, " layer")
            _dense = ls_layer['dense']
            model.add(Dense(_dense['units'], activation=_dense['activation']))
        else:
            print('没有对应层', key, value)

    if debug:
        print('compile debug')
    model.compile(loss=ls_compile['loss'], optimizer=ls_compile['optimizer'])
    if debug:
        print('fit debug')
    history = model.fit(ls_fit['x'], ls_fit['y'], epochs=ls_fit['epochs'],
                        batch_size=ls_fit['batch_size'], validation_data=ls_fit['validation_data'],
                        verbose=ls_fit['verbose'], shuffle=ls_fit['shuffle'], callbacks=ls_fit['callbacks'])

    return model, history
    # plot history


if __name__ == '__main__':
    pass