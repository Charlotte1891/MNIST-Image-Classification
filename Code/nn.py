import numpy as np
import matplotlib.pyplot as plt
import util


def softmax(x):
    z = np.max(x)
    softmax = np.exp(x - z) / np.sum(np.exp(x - z), axis=1, keepdims=True)
    return softmax


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_initial_params(input_size, num_hidden, num_output):
    W1 = np.random.standard_normal((input_size, num_hidden))
    b1 = np.zeros((1, num_hidden))
    W2 = np.random.standard_normal((num_hidden, num_output))
    b2 = np.zeros((1, num_output))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params



def forward_prop(data, labels, params):
    # set up parameters
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # forward propagation
    z = data.dot(W1) + b1
    a = sigmoid(z)
    h = softmax(a.dot(W2) + b2)

    # calculate cross-entropy loss
    n = data.shape[0]
    loss = 1 / n * np.sum(-np.multiply(labels, np.log(h)))
    return a, h, loss


def backward_prop(data, labels, params, forward_prop_func):
    # set up parameters
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    n = data.shape[0]
    a, h, loss = forward_prop_func(data, labels, params)
    delta_2 = h - labels
    dW2 = 1 / n * a.T.dot(delta_2)
    db2 = np.mean(delta_2, axis=0, keepdims=True)
    delta_1 = np.dot(delta_2, W2.T) * a * (1 - a)
    dW1 = 1 / n * data.T.dot(delta_1)
    db1 = np.mean(delta_1, axis=0, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    m = data.shape[0]
    a, h, loss = forward_prop_func(data, labels, params)
    delta_2 = h - labels
    dW2 = 1 / m * a.T.dot(delta_2) + reg * W2
    db2 = np.mean(delta_2, axis=0, keepdims=True)
    delta_1 = np.dot(delta_2, W2.T) * a * (1 - a)
    dW1 = 1 / m * data.T.dot(delta_1) + reg * W1
    db1 = np.mean(delta_1, axis=0, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func,
                           backward_prop_func):
    n = train_data.shape[0]
    num_iter = int(n / batch_size)
    for i in range(num_iter):
        batch_data = train_data[i * batch_size: (i + 1) * batch_size]
        batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]
        grads = backward_prop_func(batch_data, batch_labels, params, forward_prop_func)
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        params['W1'] -= learning_rate * dW1
        params['b1'] -= learning_rate * db1
        params['W2'] -= learning_rate * dW2
        params['b2'] -= learning_rate * db2


def nn_train(
        train_data, train_labels, dev_data, dev_labels,
        get_initial_params_func, forward_prop_func, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):
    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels,
                               learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) ==
                np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train, 'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train, 'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

    list_params = list(params.values())
    array_params = np.array(list_params)
    np.savetxt('{}_parameters.txt'.format(name), array_params, fmt='%s')

    return accuracy


def main(plot=True):
    num_epochs = 30

    # load dataset
    x_train, y_train = util.load_dataset(training_mode=True)
    m = x_train.shape[0]
    n = x_train.shape[1]
    x_train = x_train.reshape(m, n * n)
    x_test, y_test = util.load_dataset(training_mode=False)
    m = x_test.shape[0]
    n = x_test.shape[1]
    x_test = x_test.reshape(m, n * n)
    y_train = one_hot_labels(y_train)

    dev_data = x_train[0:10000, :]
    dev_labels = y_train[0:10000, :]
    train_data = x_train[10000:, :]
    train_labels = y_train[10000:, :]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data = (x_test - mean) / std
    test_labels = one_hot_labels(y_test)

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
                             lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
                             num_epochs, plot)

    return baseline_acc, reg_acc


if __name__ == '__main__':
    main()