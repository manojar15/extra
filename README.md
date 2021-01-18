
1)
import pandas as pd
data=pd.read_csv('1.csv')
rows=data.shape[0]
cols=data.shape[1]-1
h=[]
for i in range(rows):
    if data.iloc[i,cols]=="Yes":
        for j in data.iloc[i]:
            h.append(j)
        break
h.pop()
print("Initial:",h)
for i in range(1,rows):
    if data.iloc[i,cols]=="Yes":
        for j in range(cols):
            if h[j]!=data.iloc[i,j]:
                h[j]="?"
        print(i,"::::::",h)
print("Final  :",h)
   









2)
import pandas as pd
import numpy as np
data=pd.read_csv('1.csv')
rows=data.shape[0]
cols=data.shape[1]-1

concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])

spec_h=list()
for i in range(rows):
    if data.iloc[i,cols]=='Yes':
        for j in data.iloc[i]:
            spec_h.append(j)
        break
        
spec_h.pop()
gen_h = [["?" for i in range(cols)] for i in range(cols)]

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(cols):
            if h[x] != spec_h[x]:
                spec_h[x] = '?'
                gen_h[x][x] = '?'
    if target[i] == "No":
        for x in range(cols):
            if h[x] != spec_h[x]:
                gen_h[x][x] = spec_h[x]
            else:
                gen_h[x][x] = '?'
indices = [i for i, val in enumerate(gen_h) if val == ['?', '?', '?', '?', '?', '?']]
for i in indices:
    gen_h.remove(['?', '?', '?', '?', '?', '?'])
print("Final Specific_h:", spec_h, sep="\n")
print("Final General_h:", gen_h, sep="\n")

























3)
import pandas as pd
import math
from collections import Counter
from pprint import pprint
data = pd.read_csv('3.csv')

def entropy(a_list):
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)*1.0
    probs = [x / num_instances for x in cnt.values()]
    ent=sum([-prob*math.log(prob, 2) for prob in probs] )
    return ent

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [entropy, lambda x: len(x)/nobs] })
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy(df[target_attribute_name])
    return old_entropy - new_entropy

def id3(df, target_attribute_name, attribute_names):
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
    else:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        tree = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,target_attribute_name,remaining_attribute_names)
            tree[best_attr][attr_val] = subtree
        return tree
attribute_names = list(data.columns)
print("List of Attribvutes:", attribute_names)
attribute_names.remove('PlayTennis')
print("Predicting Attributes:", attribute_names)
total_entropy = entropy(data['PlayTennis'])
print("Entropy of given PlayTennis Data Set:",total_entropy)
tree = id3(data,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
pprint(tree)
















4)
from math import exp
from random import random

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
   
def transfer_derivative(output):
    return output * (1.0 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
            
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

dataset = [[2.7810836,2.550537003,0],
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.1, 10, n_outputs)
for layer in network:
        print(layer)



5)
print("\nNaive Bayes Classifier for concept learning problem")
import pandas as pd
import math
from sklearn import datasets

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return safe_div(sum(numbers),float(len(numbers)))

def stdev(numbers):
    avg = mean(numbers)
    variance = safe_div(sum([pow(x-avg,2) for x in numbers]),float(len(numbers)-1))
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)] 
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-safe_div(math.pow(x-mean,2),(2*math.pow(stdev,2))))
    final= safe_div(1 , (math.sqrt(2*math.pi) * stdev)) * exponent
    return final

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i] 
            x =inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev) 
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    accuracy = safe_div(correct,float(len(testSet))) * 100.0
    return accuracy

def main():
    df=datasets.load_iris()
    data = pd.DataFrame(df.data, columns=df.feature_names)
    data['target'] = pd.Series(df.target)
    size=len(data)
    split=int(size*0.8)
    training=data[:split]
    trainingSet=training.values.tolist()
    test = data[split:]
    testSet=test.values.tolist()
    print('Total rows ',len(data))
    print('Number of Training data: ', len(trainingSet))
    print('Number of Test Data: ',len(testSet))
    summaries= summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet) 
    actual = []
    for i in range(len(testSet)):
        vector = testSet[i]
        actual.append(vector[-1])
    print('Actual values: ',actual)
    print('Predictions:',predictions)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:',accuracy)
main()





















6)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
msg=pd.read_csv('naivetext.csv',names=['message','label'])
print('The dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted)*100)
newdata = ['this is good lab']
X_new_counts = count_vect.transform(newdata)


predictednew = clf.predict(X_new_counts)
if(predictednew==1):
    print('positive text')
else:
    print('negative text')




7)
import bayespy as bp
import numpy as np
import csv

age = {'SuperSeniorCitizen':0, 'SeniorCitizen':1, 'MiddleAged':2, 'Youth':3, 'Teen':4} 
genderEnum = {'Male':0, 'Female':1}
familyHistoryEnum = {'Yes':0, 'No':1}
dietEnum = {'High':0, 'Medium':1, 'Low':2}
lifeStyleEnum = {'Athlete':0, 'Active':1, 'Moderate':2, 'Sedetary':3}
cholesterolEnum = {'High':0, 'BorderLine':1, 'Normal':2}
heartDiseaseEnum = {'Yes':0, 'No':1}

with open('heart.csv') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    data = []
    for x in dataset:
        data.append([age[x[0]],genderEnum[x[1]],familyHistoryEnum[x[2]],dietEnum[x[3]],lifestyleEnum[x[4]],cholesterolEnum[x[5]],heartDiseaseEnum[x[6]]])
        data = np.array(data)
    N= len(data)
    
p_age = bp.nodes.Dirichlet(1.0*np.ones(5))
age = bp.nodes.Categorical(p_age, plates=(N,))
age.observe(data[:,0])

p_gender = bp.nodes.Dirichlet(1.0*np.ones(2))
gender = bp.nodes.Categorical(p_gender, plates=(N,))
gender.observe(data[:,1])

p_familyhistory = bp.nodes.Dirichlet(1.0*np.ones(2))
familyhistory = bp.nodes.Categorical(p_familyhistory, plates=(N,))
familyhistory.observe(data[:,2])

p_diet = bp.nodes.Dirichlet(1.0*np.ones(3))
diet = bp.nodes.Categorical(p_diet, plates=(N,))
diet.observe(data[:,3])

p_lifestyle = bp.nodes.Dirichlet(1.0*np.ones(4))
lifestyle = bp.nodes.Categorical(p_lifestyle, plates=(N,))
lifestyle.observe(data[:,4])

p_cholesterol = bp.nodes.Dirichlet(1.0*np.ones(3))
cholesterol = bp.nodes.Categorical(p_cholesterol, plates=(N,))
cholesterol.observe(data[:,5])

p_heartdisease = bp.nodes.Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
heartdisease = bp.nodes.MultiMixture([age, gender, familyhistory, diet, lifestyle, cholesterol],
bp.nodes.Categorical, p_heartdisease)
heartdisease.observe(data[:,6])
p_heartdisease.update()
m = 0

while m == 0:
    print("\n")
    inage=int(input('Enter Age: ' + str(ageEnum)))
    ingender=int(input('Enter Gender: ' + str(genderEnum))) 
    infamily=int(input('Enter FamilyHistory: '+ str(familyHistoryEnum))) 
    indiet=int(input('Enter dietEnum: ' +str(dietEnum)))
    inlife=int(input('Enter LifeStyle: ' + str(lifeStyleEnum)))
    inchol=int(input('Enter Cholesterol: ' + str(cholesterolEnum)))
    res = bp.nodes.MultiMixture([inage,ingender,infamily,indiet,inlife,inchol],bp.nodes.Categorical,p_heartdisease).get_moments()[0][heartDiseaseEnum['Yes']]
    print("Probability(HeartDisease) = " + str(res))
    m= int(input("Enter for Continue:0, Exit :1 "))



























8)
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
plt.figure(figsize=(7,7))
colormap = np.array(['red', 'lime', 'black'])
model = KMeans(n_clusters=3)
model.fit(X)
score1=sm.accuracy_score(y, model.labels_)
print("Accuracy of KMeans=",score1)
plt.figure(figsize=(7,7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
score2=sm.accuracy_score(y, y_cluster_gmm)
print("Accuracy of EM=",score2)
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('EM Classification')
























9)
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset=load_iris()
print("\n IRIS TARGET NAMES: \n ", iris_dataset.target_names) 
for i in range(len(iris_dataset.target_names)):
    print("\n[{0}]:[{1}]".format(i,iris_dataset.target_names[i]))
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"],random_state=0)
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)
for i in range(5):
    x = X_test[i]
    x_new = np.array([x])
    prediction= kn.predict(x_new)
    print("\n Actual : ",y_test[i],iris_dataset["target_names"][y_test[i]])
    print(" Predicted :",prediction,iris_dataset["target_names"][prediction])
print("\n TEST SCORE[ACCURACY]: \n",kn.score(X_test, y_test)*100)
x_new = np.array([[5, 2.9, 1, 0.2]])
print("\n XNEW \n",x_new)
prediction = kn.predict(x_new)
print("\n Predicted target value: \n",prediction)
print("\n Predicted feature name: \n",iris_dataset["target_names"][prediction])





10)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
import statsmodels.api as sm

x=[i/5.0 for i in range(30)]
y = [1,2,1,2,1,1,3,4,5,4,5,6,5,6,7,8,9,10,11,11,12,11,11,10,12,11,11,10,9,13]
lowess = sm.nonparametric.lowess(y, x)
lowess_x = list(zip(*lowess))[0]
lowess_y = list(zip(*lowess))[1]
f = interp1d(lowess_x, lowess_y,bounds_error=False)
xnew = [i/10.0 for i in range(100)]
ynew = f(xnew)
plt.plot(x, y, 'o')
plt.plot(lowess_x,lowess_y, '*')
plt.plot(xnew, ynew, '-')
plt.show()

