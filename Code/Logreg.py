import numpy as np
import matplotlib.pyplot as plt
import util

# number of class
K = 10

def main(save_path):
    # load dataset
    x_train, y_train = util.load_dataset(training_mode=True)
    m = x_train.shape[0]
    n = x_train.shape[1]
    x_train = x_train.reshape(m, n * n)
    index = np.random.choice(x_train.shape[0], 10000, replace=False)
    x_train = x_train[index]
    y_train = y_train[index]
    x_test, y_test = util.load_dataset(training_mode=False)
    m = x_test.shape[0]
    n = x_test.shape[1]
    x_test = x_test.reshape(m, n * n)

    dev_data = x_train[0:2000, :]
    dev_labels = y_train[0:2000]
    train_data = x_train[2000:, :]
    train_labels = y_train[2000:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std
    test_data = (x_test - mean) / std
    test_labels = y_test

    # Train a logistic regression classifier
    clf = SoftmaxRegression()
    theta = clf.fit(train_data, train_labels)
    train_acc = clf.get_accuracy(train_data, train_labels)
    theta = clf.fit(dev_data, dev_labels)
    dev_acc = clf.get_accuracy(dev_data, dev_labels)

    # Use np.savetxt to save predictions on eval set to save_path
    pred_value = clf.predict(test_data)
    acc = clf.get_accuracy(test_data, test_labels)
    print('training set accuracy is %s' % train_acc)
    print('dev set accuracy is %s' % dev_acc)
    print("model accuracy is %s" % acc)
    np.savetxt(save_path, pred_value)


class SoftmaxRegression:

    def __init__(self, max_iter=10000, eps=1e-4, alpha=0.01, theta_0=None, verbose=True):
        self.theta = theta_0
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, x, y):
        # Initialize theta
        if self.theta is None:
            self.theta = np.zeros((x.shape[1], K))
        Lambda = 0.1
        n_iter = 1
        while n_iter < self.max_iter:
            prev_theta = np.copy(self.theta)
            self.theta -= self.alpha * self.gradient(self.theta, x, y)
            loss = self.loss_func(self.theta, x, y)
            # ridge regression
            # loss = self.ridge_loss(self.theta, x, y, Lambda)
            # lasso regression
            # loss = self.lasso_loss(self.theta, x, y, Lambda)

            if self.verbose:
                print('loss of iteration {} is {}'.format(n_iter, round(loss, 3)))

            if np.sum(np.abs(self.theta - prev_theta)) < self.eps:
                break

            n_iter += 1

        return self.theta


    def predict(self, x):
        """Return predicted probabilities given new inputs x.
        """
        pred_prob = self.softmax(x.dot(self.theta))
        pred_value = np.argmax(pred_prob, axis=1)
        if self.verbose:
            return pred_value

    # Accuracy function
    def get_accuracy(self, x, y):
        acc = np.mean(self.predict(x) == y)
        return acc

    def gradient(self, theta, x, y):
        m = x.shape[0]
        hx = self.softmax(x.dot(theta))
        grad = 1/m * x.T.dot(hx-self.one_hot_labels(x, y))
        return grad

    # Cross-entropy loss
    def loss_func(self, theta, x, y):
        m = x.shape[0]
        hx = self.softmax(x.dot(theta))
        # cross-entropy loss can be expressed as one_hot * np.log(hx)
        loss = 1/m * np.sum(-np.log(hx[np.arange(m), y]))
        return loss

    # with Lasso regularization
    def lasso_loss(self, theta, x, y, Lambda):
        m = x.shape[0]
        hx = self.softmax(x.dot(theta))
        loss = 1/m * np.sum(-np.log(hx[np.arange(m), y])) + Lambda/(2 * m) * np.sum(np.abs(theta))
        return loss

    # with Ridge regularization
    def ridge_loss(self, theta, x, y, Lambda):
        m = x.shape[0]
        hx = self.softmax(x.dot(theta))
        loss = 1/m * np.sum(-np.log(hx[np.arange(m), y])) + Lambda/(2 * m) * np.sum(theta ** 2)
        return loss

    @staticmethod
    def one_hot_labels(x, y):
        m = x.shape[0]
        one_hot_labels = np.zeros((m, K))
        one_hot_labels[np.arange(m), y] = 1
        return one_hot_labels

    @staticmethod
    def softmax(x):
        x = x.astype(np.float128)
        z = np.max(x)
        softmax = np.exp(x-z) / np.sum(np.exp(x-z), axis=1, keepdims=True)
        return softmax


if __name__ == '__main__':
    main(save_path='logreg_pred.txt')