#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
import numpy as np
import seaborn as sns


# In[2]:


# important constants
K = 9
n_folds = 5


# In[3]:


def calc_euclidean_distance(row1, row2):
    if (len(row1) != len(row2)):
        raise Exception("Something is wrong with your data! Both rows are not the same size")
    distance = 0.0
    for i in range(len(row1)-1):
        # assume that last row is value we want to predict
        # use for loop so that we can scale for higher dimensions as well
        distance = distance + (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# In[4]:


def get_most_similar_neighbors(train, test, k):
    distances = []
    for row in train:
        curr_dist = calc_euclidean_distance(row, test)
        distances.append((row, curr_dist))
    distances.sort(key = lambda x : x[1]) #sorting by distance
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


# In[5]:


def make_prediction(train, test, k):
    neighbors = get_most_similar_neighbors(train, test, k)
    last_row = [curr[len(curr)-1] for curr in neighbors]
    model_prediction = max(set(last_row), key=last_row.count)
    return model_prediction


# In[6]:


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[{}] ==> {}'.format(value, i)) #PRINT MAPPINGS 
        #print(lookup[value])
    for row in dataset:
        row[column] = int(lookup[row[column]])
    #return lookup

def str_column_to_float(dataset, column):
    for row in dataset:
        if (type(row[column]) == str):
            row[column] = int(row[column])


# In[7]:


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# In[8]:


## Model Evaluation Loop
def cross_validation_split(dataset, n_folds):
    split = list()
    copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(copy))
            fold.append(copy.pop(index))
        split.append(fold)
    return split

def evaluate_algorithm(dataset, n_folds, K):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    
    for fold in folds:
        train = list(folds)
        train.remove(fold)
        train = sum(train, [])
        test = list()
        
        for row in fold:
            row_copy = list(row)
            test.append(row_copy)
        #print(train)
        #print("\n", test, " ", len(test), " ", test[1])
        
        predicted = []
        for i in range(len(test)):
            curr_prediction = make_prediction(train, test[i], K)
            #print(curr_prediction)
            predicted.append(curr_prediction)
            
        actual = [row[-1] for row in fold]
        #print(predicted)
        #print(actual)
        
        accuracy = accuracy_metric(actual, predicted)
        
        scores.append(accuracy)
    return scores, predicted


# In[9]:


def plot(X, y, h):
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#9bc2cf'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    acc, Z = evaluate_algorithm(np.c_[xx.ravel(), yy.ravel()], n_folds, K)
    print("FINISHED: ", Z.shape)
    
    Z = Z.reshape(xx.shape)
    
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.show()


# In[ ]:


def KNN_IRIS():
    dataset = pd.read_csv('iris.csv')
    dataset = dataset.values.tolist()
    
    test = pd.read_csv('iris_test.csv')
    test = test.values.tolist()
    
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    
#     dataset = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                      columns= iris['feature_names'] + ['target'])
#     dataset = dataset.values.tolist()

#     print(X)
#     print(y)
    
#     dataset = dataset.values.tolist()
#     test = test.values.tolist()

    str_column_to_int(dataset, len(dataset[0])-1)
    str_column_to_int(test, len(test[0])-1)
    
#     answers = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']
    
#     #print(dataset)
    
#     for i in range(len(dataset[0])):
#         str_column_to_float(iris, i)
#         str_column_to_float(test, i)
    


    scores, predictions = evaluate_algorithm(dataset, n_folds, K)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    h = .02  # step size in the mesh

    plot(X,y,h)
    
    
#     right = 0
#     wrong = 0
#     x = 1
#     for x in range(len(test)):
#         curr_prediction = make_prediction(dataset, test[x], K)
#         print("Prediction: ", answers[curr_prediction], " , Actual: ", answers[test[x][4]])
#         if (answers[curr_prediction] == answers[test[x][4]]):
#             right += 1
#         else:
#             wrong += 1
#         x += 1
        
#     print(right / (right + wrong))

KNN_IRIS()


# In[ ]:




