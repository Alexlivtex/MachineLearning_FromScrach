from LinearRegression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    x, y = make_regression(n_samples=100, n_features=10, noise=20)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    n_samples, n_features = np.shape(x_train)
    print(n_samples, n_features)

    model = LinearRegression(iterations_count=100, learningRate=0.001, lambdaRate=0.1)
    model.training(x_train, y_train)

if __name__ == "__main__":
    main()