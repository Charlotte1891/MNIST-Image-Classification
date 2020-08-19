import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import plot_model
import util

# load train and test dataset
def load_dataset():
    x_train, y_train = util.load_dataset(training_mode=True)
    m = x_train.shape[0]
    n = x_train.shape[1]
    x_train = x_train.reshape(m, n, n, 1)
    x_test, y_test = util.load_dataset(training_mode=False)
    m = x_test.shape[0]
    n = x_test.shape[1]
    x_test = x_test.reshape(m, n, n, 1)
    # one hot labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def norm_preprocess(train, test):
    train_norm = train.astype('float32') / 255.0
    test_norm = test.astype('float32') / 255.0
    return train_norm, test_norm


def build_model():
    model = Sequential()
    # model 1
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model 2
    # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(10, activation='softmax'))
    # opt = SGD(lr=0.01, momentum=0.9)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(x, y, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    i = 0
    for train, test in kfold.split(x):
        i += 1
        model = build_model()
        x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print('> %.3f' % (test_acc * 100.0))
        scores.append(test_acc)
        histories.append(history)
        plot_model(model, to_file='CNN_easy_model_{}.png'.format(i), show_shapes=True)
    return scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title('Cross Entropy Loss')
        ax1.plot(histories[i].history['loss'], color='blue', label='train')
        ax1.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        ax2.set_title('Classification Accuracy')
        ax2.plot(histories[i].history['accuracy'], color='blue', label='train')
        ax2.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()


def summarize_performance(scores):
    print('Accuracy: mean=%3.f, std=%3.f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()

def main():
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, x_test = norm_preprocess(x_train, x_test)
    scores, histories = evaluate_model(x_train, y_train)
    summarize_diagnostics(histories)
    summarize_performance(scores)


if __name__ == '__main__':
    main()