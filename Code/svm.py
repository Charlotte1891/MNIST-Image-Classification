import util
import datetime as dt
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix,

    cm - confusion matrix
    """
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('svm.png')
    plt.show()


def main():
    # load dataset
    x_train, y_train = util.load_dataset(training_mode=True)
    m = x_train.shape[0]
    n = x_train.shape[1]
    x_train = x_train.reshape(m, n * n)
    x_test, y_test = util.load_dataset(training_mode=False)
    m = x_test.shape[0]
    n = x_test.shape[1]
    x_test = x_test.reshape(m, n * n)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # build SVM model
    svm = SVC(C=5, gamma=0.05)

    start_time = dt.datetime.now()
    svm.fit(x_train, y_train)
    end_time = dt.datetime.now()
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))

    # predict
    expected = y_test
    predicted = svm.predict(x_test)

    # confusion matrix
    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)
    plot_confusion_matrix(cm)

    # get accuracy
    print("Training Accuracy={}".format(metrics.accuracy_score(y_train, svm.predict(x_train))))
    print("Testing Accuracy={}".format(metrics.accuracy_score(y_test, svm.predict(x_test))))


if __name__ == '__main__':
    main(save_path='logreg_pred.txt')