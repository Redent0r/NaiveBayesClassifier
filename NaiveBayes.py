# Saul Paredes
# sbp14c
# python3 NaiveBayes.py breast_cancer.train breast_cancer.test 

import sys
import math

def run():
    argv = sys.argv

    if (len(argv) < 3):
        print("Sorry, wrong length of filenames")
        sys.exit()

    train = [] # train dataset
    test = [] # test dataset

    trainLabels = set()
    testLabels = {}

    attributes = set()

    # build training dataset
    fileTrain = open(argv[1],"r")
    for line in fileTrain:
        field = line.split(" ")
        label = field[0]

        trainLabels.add(label)

        trainingInstance = {}
        trainingInstance["label"] = label

        attr = {}
        for i in range(1, len(field)):
            index = field[i].split(":")[0]
            value = int(field[i].split(":")[1])
            attr[index] = value
            attributes.add(index)

        trainingInstance["attributes"] = attr;
        train.append(trainingInstance)
    #print("train: ", train)
    
    #print("labels: ", sorted(trainLabels))
    #print("attributes: ", sorted(attributes))

    # train
    means = {} # sample mean
    attrToLabel = {} # N
    variance = {} # sample variance

    for attribute in attributes: # initialize tables
        means[attribute] = {}
        attrToLabel[attribute] = {}
        variance[attribute] = {}
        for label in trainLabels:
            means[attribute][label] = 0
            attrToLabel[attribute][label] = 0
            variance[attribute][label] = 0

    for trainInstance in train: # calculate sum(x) and N
        label = trainInstance["label"]
        #for attribute in trainInstance["attributes"]:
        for attribute in attributes:
            if attribute in trainInstance["attributes"]: means[attribute][label] += trainInstance["attributes"][attribute]
            attrToLabel[attribute][label] += 1

    for attr in means: # calculate mean = sum(x)/N
        for label in means[attr]:
            if(attrToLabel[attr][label] != 0): means[attr][label] /= attrToLabel[attr][label]

    #print("means: ", means)
    
    for trainInstance in train: # sum((x - mean)^2)                                                              
        label = trainInstance["label"]
        #for attribute in trainInstance["attributes"]:
        for attribute in attributes:
            if attribute in trainInstance["attributes"]: variance[attribute][label] += ((trainInstance["attributes"][attribute] - means[attribute][label]) ** 2)
            else: variance[attribute][label] += ((0 - means[attribute][label]) ** 2) 

    for attr in variance: # calculate variance = sum((x - mean)^2)/N-1                                         
        for label in variance[attr]:
            if attrToLabel[attr][label] > 2: variance[attr][label] /= (attrToLabel[attr][label] - 1)

    #print("variance: ", variance)

    # from this point on, we assume that the labels are "+1" and "-1"
    # so we can build the confusion matrices

    # predict train                                                               
    confusionTrain = {"TP":0, "FN":0, "FP":0, "TN":0}
    for trainInstance in train:
        prob = {}
        attributes = trainInstance["attributes"]
        for attribute in attributes:
            for label in trainLabels:
                #print("attr", attribute)
                #print("label", label)
                if label not in prob: prob[label] = normalProb(attributes[attribute], means[attribute][label], variance[attribute][label]) # equals                                                                                         
                else: prob[label] *= normalProb(attributes[attribute], means[attribute][label], variance[attribute][label]) # multiply times itself                                                                                         
        trueLabel = trainInstance["label"]
        #print(prob)                                                                                                  
        predictedLabel = max(prob, key=prob.get) # get the key that has highest probability
        #print("true = ", trueLabel, " | predicted: ", predictedLabel)                                                
        if(trueLabel == predictedLabel): # true                                                                       
            if(predictedLabel == "+1"): confusionTrain["TP"] += 1 # positive                                          
            else: confusionTrain["TN"] += 1 # negative                                                                
        else: # false                                                                                                 
            if(predictedLabel == "+1"): confusionTrain["FP"] += 1 # positive                                          
            else: confusionTrain["FN"] += 1 # negative                                                                

    print(confusionTrain["TP"], " ", confusionTrain["FN"], " ", confusionTrain["FP"], " ",confusionTrain["TN"])

    # build testing dataset                                                                                    
    fileTest = open (argv[2], "r")
    for line in fileTest:
        field = line.split(" ")
        label = field[0]
        testingInstance = {}
        testingInstance["label"] = label
        attr = {}
        for i in range(1, len(field)):
            index = field[i].split(":")[0]
            value = int(field[i].split(":")[1])
            attr[index] = value

        testingInstance["attributes"] = attr;
        test.append(testingInstance)

    # predict test
    confusionTest = {"TP":0, "FN":0, "FP":0, "TN":0}
    for testInstance in test:
        prob = {}
        #for label in trainLabels: prob[label] = 0 # inital probability
        attributes = testInstance["attributes"]
        for attribute in attributes:
            for label in trainLabels:
                if label not in prob: prob[label] = normalProb(attributes[attribute], means[attribute][label], variance[attribute][label]) # equals
                else: prob[label] *= normalProb(attributes[attribute], means[attribute][label], variance[attribute][label]) # multiply times itself
        trueLabel = testInstance["label"]
        #print(prob)
        predictedLabel = max(prob, key=prob.get) # get the key that has highest probability
        #print("true = ", trueLabel, " | predicted: ", predictedLabel)
        if(trueLabel == predictedLabel): # true
            if(predictedLabel == "+1"): confusionTest["TP"] += 1 # positive
            else: confusionTest["TN"] += 1 # negative
        else: # false
            if(predictedLabel == "+1"): confusionTest["FP"] += 1 # positive
            else: confusionTest["FN"] += 1 # negative

    print(confusionTest["TP"], " ", confusionTest["FN"], " ", confusionTest["FP"], " ",confusionTest["TN"])

def normalProb(x, mean, variance):
    #print("calculating x = ", x ,", mean = ", mean, ", variance = ", variance)
    if variance == 0: variance = 0.001
    denom = math.sqrt((2 *math.pi * variance))
    num = math.exp(-(float(x)-float(mean))**2/(2*variance))
    return num/denom
    
if __name__ == '__main__':
    run()
