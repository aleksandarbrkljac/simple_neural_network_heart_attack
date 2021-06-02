from __future__ import print_function

from abc import abstractmethod
import math
import random
import copy
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from matplotlib import pyplot
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
import os.path
from os import path
import imblearn.over_sampling as RandomOverSampling
from sklearn import preprocessing


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):  # x is an array of scalars
        pass

    @abstractmethod
    def backward(self, dz):  # dz is a scalar
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] is input, x[1] is weight

    def forward(self, x):
        self.x = x
        return self.x[0] * self.x[1]

    def backward(self, dz):
        return [dz * self.x[1], dz * self.x[0]]


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x is in an array of inputs

    def forward(self, x):
        self.x = x
        return sum(self.x)

    def backward(self, dz):
        return [dz for xx in self.x]


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        return 1. / (1. + math.exp(-x))


class TanhNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._tanh(self.x)

    def backward(self, dz):
        return dz * (1. - (self._tanh(self.x))**2)

    def _tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0.)

    def _relu(self, x):
        return max(0., x)


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs
        self.multiply_nodes = []  # for inputs and weights
        self.sum_node = SumNode()  # for sum of inputs*weights

        for n in range(n_inputs):  # collect inputs and corresponding weights
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]  # init input weights
            self.multiply_nodes.append(mn)

        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        elif activation == "tanh":
            self.activation_node = TanhNode()
        else:
            raise RuntimeError(
                'Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x is a vector of inputs
        x = copy.copy(x)
        x.append(1.)  # for bias

        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))

        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        dx = []
        b = dz[0] if type(dz[0]) == float else sum(dz)

        b = self.activation_node.backward(b)
        b = self.sum_node.backward(b)
        for i, bb in enumerate(b):
            dw.append(self.multiply_nodes[i].backward(bb)[1])
            dx.append(self.multiply_nodes[i].backward(bb)[0])

        self.gradients = dw
        return dx

    def update_weights(self, learning_rate, momentum):
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_gradient = self.gradients[i]
            delta = learning_rate*mean_gradient + \
                momentum*self.previous_deltas[i]
            self.previous_deltas[i] = delta
            self.multiply_nodes[i].x[1] -= delta

        self.gradients = []


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        self.neurons = []
        # construct layer
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x is a vector of "n_inputs" elements
        layer_output = []
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz is a vector of "n_neurons" elements
        b = []
        for idx, neuron in enumerate(self.neurons):
            neuron_dz = [d[idx] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            b.append(neuron_dz[:-1])

        return b  # b is a vector of "n_neurons" elements

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        # construct neural network
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x is a vector which is an input for neural net
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)

        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)

        return next_layer_dz

    def update_weights(self, learning_rate, momentum):
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate, momentum, nb_epochs, shuffle=False, verbose=0):
        assert len(X) == len(Y)

        hist = []
        for epoch in range(nb_epochs):
            if shuffle:
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            for x, y in zip(X, Y):
                # forward pass to compute output
                pred = self.forward(x)
                # compute loss
                grad = 0.0
                for o, t in zip(pred, y):
                    total_loss += (t - o) ** 2.
                    grad += -(t - o)
                # backward pass to compute gradients
                self.backward([[grad]])
                # update weights with computed gradients
                self.update_weights(learning_rate, momentum)

            hist.append(total_loss)
            if verbose == 1:
                print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)


def normalize(data):
    x_min = min(data)
    x_max = max(data)
    retval = []
    for x in data:
        retval.append((x-x_min)/(x_max-x_min))
    return retval

def explore(X,Y):
    encodx = pd.get_dummies(X)
    types = encodx.columns.values
    sumlist = []
    for i in types:
        sum = 0
        #Koliko 1 ima u enkodovanom nizu, koje daju 1 na izlazu
        for s in zip(Y.values, encodx[i].values):
            if s[0] == 1 and s[1] == 1:
                sum += 1
        sumlist.append(sum)
    x_pos = [i for i, _ in enumerate(types)]
    pyplot.style.use('ggplot')
    pyplot.bar(x_pos, sumlist)
    pyplot.xticks(x_pos, types)
    pyplot.show()

def oversample(data):
    max_size = data['stroke'].value_counts().max()
    lst = [data]
    for class_index, group in data.groupby('stroke'):
        lst.append(group.sample(max_size-len(group), replace=True))
    frame_new = pd.concat(lst)
    return frame_new

if __name__ == '__main__':
    

    data = pd.read_csv(
        "C:\\Users\\Andrea\\Desktop\\OneDrive_2021-05-26\\Kolokvijum 2\\dataset.csv")
   
    data.dropna(inplace=True)  #Remove NaN rows
    #Oversampling
    #print(data['stroke'].value_counts())
    data=oversample(data)
    #print(data['stroke'].value_counts())
    explore(data.ever_married,data.stroke)
    genderdata = pd.get_dummies(data.gender) #Gender OneHotEncoding data
    work_type_encoded = pd.get_dummies(data.work_type) #WorkType OneHotEncoding data
    smoking_status_encoded = pd.get_dummies(data.smoking_status) #SmokingStatus OneHotEncoding data
    ever_married_data = data['ever_married'].replace({'Yes': 1, 'No': 0}, inplace=False)
    Y = data['stroke'].values
    #DataNormalization
    age_data_normalized = normalize(data.age.values)
    avg_glucose_level_normalized = normalize(data.avg_glucose_level.values)
    bmi_data_normalized = normalize(data.bmi.values)
    #Create input data
    X = list(zip(data.heart_disease, age_data_normalized, data.hypertension, avg_glucose_level_normalized, bmi_data_normalized,
                 ever_married_data.values,
                 work_type_encoded.Govt_job, work_type_encoded.Never_worked, work_type_encoded.Private, work_type_encoded['Self-employed'].values,
                 work_type_encoded.children,
                 smoking_status_encoded['Unknown'].values, smoking_status_encoded['formerly smoked'].values, smoking_status_encoded['never smoked'].values, 
                 smoking_status_encoded['smokes'].values))
    
    #Oversampling zbog malog broja True izlaza
    #oversample = RandomOverSampling.RandomOverSampler(sampling_strategy=0.5)
    #X_over, Y_over = oversample.fit_resample(X, Y)

    #Splitovanje dataseta na train/test (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    
    #Input/Output za trening
    X_trainn = [list(d) for d in X_train]
    y_trainn = [[float(s)] for s in y_train]
    #Input/Output za test
    X_testt = [list(d) for d in X_test]
    y_testt = [[float(s)] for s in y_test]
    #print(f"Training target statistics: {Counter(y_train)}")
    #print(f"Testing target statistics: {Counter(y_test)}")

    if(path.exists("C:\\Users\\Andrea\\Desktop\\OneDrive_2021-05-26\\Kolokvijum 2\\nn.p")):
        nn = pickle.load(open(
            "C:\\Users\\Andrea\\Desktop\\OneDrive_2021-05-26\\Kolokvijum 2\\nn.p", "rb"))
    else:
        nn = NeuralNetwork()
        nn.add(NeuralLayer(15, 18, 'sigmoid'))
        nn.add(NeuralLayer(18, 10, 'tanh'))
        nn.add(NeuralLayer(10, 1, 'sigmoid'))
        history = nn.fit(X_trainn, y_trainn, learning_rate=0.1,
                            momentum=0.3, nb_epochs=100, shuffle=True, verbose=1)
        pyplot.plot(history)
        pyplot.show()
        pickle.dump(nn, open("C:\\Users\\Andrea\\Desktop\\OneDrive_2021-05-26\\Kolokvijum 2\\nn.p", "wb"))

    true_positiv = 0
    true_negativ = 0
    false_positive = 0
    false_negative = 0
    #testiranje
    for idx, xx in enumerate(X_testt):
       #print(xx)
       #print(y_test[idx])
        p = nn.predict(xx)
        # print(p)
        if y_testt[idx] == [1.0]:
            if p[0] > 0.5:
                true_positiv += 1
            else:
                false_negative += 1
        else:  # stvarno je bilo false
            if p[0] < 0.5:  # predividnjeno je false
                true_negativ += 1
            else:
                false_positive += 1

    print(f"TP: {true_positiv}")
    print(f"TN: {true_negativ}")
    print(f"FP: {false_positive}")
    print(f"FN: {false_negative}")
    precision = true_positiv/(true_positiv+false_positive)
    recall = true_positiv/(true_positiv+false_negative)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    f1score= precision*recall / (precision+recall)
    print(f"F1 Score: {f1score}")