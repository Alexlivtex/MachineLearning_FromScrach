import numpy as np
from MachineLearning_Imp.utils.misc import euclidean_distance

class KNN(object):
    def __init__(self, k=5):
        self.v_k = k

    def getMaxCount(self, labels):
        """
        Get the maxmount of labels
        """
        counts = np.bincount(labels.astype('int'))
        return counts.argmax()

    def predict(self, x_train, y_train, x_test):
        y_predict = np.empty(x_test.shape[0])
        for index, value in enumerate(x_test):
            # ���ؾ���δ֪�������K������±꣬��������
            idx = np.argsort(euclidean_distance(value, x) for x in x_train)[:self.v_k]
            # ��ȡ��Щ���Ӧ�����
            neighbor = np.array([y_train[i] for i in idx])
            # ���δ֪�����������
            y_predict[index] = self.getMaxCount(neighbor)
        return y_predict