import matplotlib
import numpy
from sklearn import datasets, linear_model, discriminant_analysis, cross_decomposition, model_selection
from enum import Enum
class Model(Enum):
    LinearRegression = linear_model.LinearRegression()
    Ridge = linear_model.Ridge()
    Lasso = linear_model.Lasso()

def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target,
                                            test_size=0.25, random_state=0)

def test_LinearRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_, regr.intercept_))
    print("Residual sum of squares : %.2f" % (numpy.mean(regr.predict(X_test)-y_test)**2))
    print('Score: %.2f'% regr.score(X_test, y_test))


def test_Ridge(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares : %.2f" % (numpy.mean(regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

def test_Lasso(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares : %.2f" % (numpy.mean(regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

def test_linear_model(model, *data):
    X_train, X_test, y_train, y_test = data
    regr = model.value
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares : %.2f" % (numpy.mean(regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))


def test_linear_model2(model, *data):
    X_train, X_test, y_train, y_test = data
    regr = model
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares : %.2f" % (numpy.mean(regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

def test_linear_model3(model, *data):
    X_train, X_test, y_train, y_test = data
    regr = model
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %s' % (regr.coef_, regr.intercept_))
    # print("Residual sum of squares : %.2f" % (numpy.mean(regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

def load_data2():
    iris = datasets.load_iris()
    X_train=iris.data
    y_train = iris.target

    return model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    # test_LinearRegression(X_train, X_test, y_train, y_test)
    # test_Ridge(X_train, X_test, y_train, y_test)
    # test_Lasso(X_train, X_test, y_train, y_test)

    # test_linear_model(Model.Ridge, X_train, X_test, y_train, y_test)
    test_linear_model2(linear_model.LinearRegression(), X_train, X_test, y_train, y_test)
    test_linear_model2(linear_model.Ridge(), X_train, X_test, y_train, y_test)
    test_linear_model2(linear_model.Lasso(), X_train, X_test, y_train, y_test)

    test_linear_model2(linear_model.ElasticNet(), X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = load_data()
    test_linear_model3(linear_model.LogisticRegression(), X_train, X_test, y_train, y_test)