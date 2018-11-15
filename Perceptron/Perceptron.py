import operator
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import time
import threading

trainingInterval = 0.000001

weights_array = []
bias_array = []

def genData(nSamples, nFeatures):
    data, target = make_blobs(n_samples=nSamples, centers=2, n_features=nFeatures, random_state=1)
    for index in range(len(target)):
        if target[index] == 0:
            target[index] = 1
        elif target[index] == 1:
            target[index] = -1
    return data, target


def caculate(row, label):
    global weights, b
    result = 0
    for index in range(len(row)):
        result += weights[index] * row[index]
    result += b
    result *= label
    return result

def update(row, label, stepLength = 0.1):
    global weights, b
    for i in range(len(row)):
        weights[i] += label * row[i] * stepLength
    b += label * stepLength

def PerceptronClassify(trainData, trainLabels):
    global weights, b, solutionFound, trainingInterval
    solutionFound = False
    nSamples = trainData.shape[0]
    nFeatures = trainData.shape[1]

    weights = [0]*nFeatures
    b = 0
    while not solutionFound:
        time.sleep(trainingInterval)
        for i in range(nSamples):
            if caculate(trainData[i], trainLabels[i]) <= 0:
                print(weights)
                print(b)
                print(trainData[i])
                print(trainLabels[i])
                update(trainData[i], trainLabels[i])
                weights_array.append(weights)
                bias_array.append(b)
                break
            elif i == nSamples - 1:
                print(weights)
                print(b)
                solutionFound = True


def updateFig(aux, fig):
    x = np.linspace(1, 100, 20)
    y = x * 2 + 3
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    plt.ion()
    for i in range(10):
        y = x * i * 0.1 + i
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(x, y)
        plt.pause(1)

def drawLearningProcess(x_data, y_data):
    global solutionFound, trainingInterval, weights, b, weights_array
    X = x_data
    y = y_data
    # scatter plot, dots colored by class value
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key + 1])
    print([weights[1], b])
    print([weights[0], 0])
    ax.plot([weights[1], 0],[weights[0], b], 'k-')
    plt.show()

if __name__ == '__main__':
    data_X, data_Y = genData(600, 2)
    PerceptronClassify(data_X, data_Y)
    drawLearningProcess(data_X, data_Y)

