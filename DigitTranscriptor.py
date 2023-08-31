import random

import pandas as pd
import numpy as np
import matplotlib as plt
import pickle as pk
import os as os

df_test = pd.read_csv(os.getcwd()+'\\data\\mnist_test.csv')

test = np.array(df_test).T

m, n = test.shape

x_test = test[1:n]/255.
y_test = test[0]


def paraminit():
    return pk.load(file=open("data/vars.txt", "rb"))


def softmax(Z):
    A = Z.T
    for i in range(np.size(A, axis=0)):
        for j in range(np.size(A[i])):
            A[i][j] = np.e**A[i][j] / sum(np.exp(A[i]))
    return A.T


def ReLU(Z):
    x,y = Z.shape
    for i in range(x):
        for j in range(y):
            if Z[i][j] < 0:
                Z[i][j] = 0
    return Z


def forward_prop(W1, W2, b1, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def get_predictions(A2):
    return np.argmax(A2, 0)


def make_prediction(W1, W2, b1, b2, X):
    _, _, _, A2 = forward_prop(W1, W2, b1, b2, X)
    prediction = get_predictions(A2)
    return prediction
#Renee WAS HERE


def test_prediction(index, Xset, Yset, W1, W2, b1, b2):
    prediction = make_prediction(W1, W2, b1, b2, Xset[:, index, None])
    label = Yset[index]
    print("Prediction:", prediction)
    print("Actual:", label)
    return prediction == label

if __name__ == "__main__":
    paramarr = pk.load(file=open("data/vars2.txt", "rb"))
    W1 = paramarr[0]
    W2 = paramarr[1]
    b1 = paramarr[2]
    b2 = paramarr[3]
    k = 0
    for i in range(0, m):
        if test_prediction(i, x_test, y_test, W1, W2, b1, b2):
            k = k+1
    acc = k/m
    print("Accuracy: "+(acc*100).__str__()+"%")



