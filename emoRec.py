#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for training, evaluating and using classifiers to predict emotions
from 3D facial landmarks.
"""

__version__ = "0.7.3"
__author__ = (
    "Robin Larsson, Daniel Norell,"
    " Sebastian Nödtvedt & Robert Rosborg"
)

import sys

import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm, colors, pyplot

from fileReader import readLm3Files
from featCalc import calcFeats, calcSimpleFeats
import argvParser

labelMap = {
    "ANGER": 0,
    "DISGUST": 1,
    "FEAR": 2,
    "HAPPY": 3,
    "SADNESS": 4,
    "SURPRISE": 5,
}

reverseLabelMap = {
    0: "ANGER",
    1: "DISGUST",
    2: "FEAR",
    3: "HAPPY",
    4: "SADNESS",
    5: "SURPRISE",
}


class DataPackage:
    """
    Stores data for the machine learning algorithms.
    """

    def __init__(self, dataDir=None, simple=False):
        """
        Initialize an object by reading files in dataDir, if given.
        """
        if dataDir:
            data, labels, imageNames = readLm3Files(dataDir)

            self.landmarks = np.array(data)
            self.labels = [labelMap[label] for label in labels]
            self.imageNames = imageNames

            if simple:
                self.features = calcSimpleFeats(self.landmarks,normalize=False)
            else:
                self.features = calcFeats(self.landmarks)
        else:
            self.landmarks = []
            self.labels = []
            self.imageNames = []
            self.features = []

        self.nClasses = 6
        self.nFiles = 0  # Per class


def splitData(feats, percentSplit=0.8):
    """
    Split the data into a training and a testing part.

    Args:
        feats(numpy.ndarray): 2D array containing features.  
        percentsplit(float): Float representing how many percentage to perform the training on.

    Returns:
         Tuple: A tuple containing data,labels and image names for both training and testing.
    """
    nClasses = feats.nClasses

    data = [[] for i in range(nClasses)]
    labels = [[] for i in range(nClasses)]
    imageNames = [[] for i in range(nClasses)]

    for i in range(len(feats.features)):
        index = feats.labels[i]

        data[index].append(feats.features[i])
        labels[index].append(index)
        imageNames[index].append(feats.imageNames[i])

    feats.nFiles = len(min(data, key=len))
    nTrainSamples  = int(feats.nFiles * percentSplit)
    nTestSamples   = feats.nFiles - nTrainSamples

    trainData = np.zeros((nClasses * nTrainSamples, feats.features.shape[1]))
    trainLabels = []
    trainImageNames = []

    if nTestSamples > 0:
        testData = np.zeros((nClasses * nTestSamples, feats.features.shape[1]))
        expectedLabels = []
        testImageNames = []
    else:
        testData = None
        expectedLabels = None
        testImageNames = None

    for i in range(feats.nClasses):
        iTrain = i * nTrainSamples
        iTest  = i * nTestSamples

        trainData[iTrain : iTrain + nTrainSamples] = data[i][:nTrainSamples]
        trainLabels += labels[i][:nTrainSamples]
        trainImageNames += imageNames[i][:nTrainSamples]

        if nTestSamples > 0:
            testData[iTest : iTest + nTestSamples] = (data[i][nTrainSamples
                                                              : nTrainSamples
                                                              + nTestSamples])
            expectedLabels += labels[i][nTrainSamples
                                        : nTrainSamples + nTestSamples]
            testImageNames += imageNames[i][nTrainSamples
                                            : nTrainSamples + nTestSamples]
   
    return (trainData, trainLabels, trainImageNames,
            testData, expectedLabels, testImageNames)


def setupKNN(trainData, trainLabels, nNeighbors=10):
    """
    Setup and train a classifier using k-NN.

    Args:
        trainData(numpy.ndarray): 2D array containing training data.
        trainLabels(list): List containing training labels.
        nNeighbors(int): Int representing the number of neighbors on which to use KNN.

    Returns:
         KNeighborsClassifier: A KNeighborsClassifier containing necessary information 
         about the classifier.
    """
    
    clfKNN = KNeighborsClassifier(nNeighbors, weights='distance')
    clfKNN.fit(trainData, trainLabels)
    
    return clfKNN


def setupSVM(trainData, trainLabels):
    """
    Setup and train a classifier using SVM.

    Args:
        trainData(numpy.ndarray): 2D array containing training data.
        trainLabels(list): List containing training labels.

    Returns:
         CalibratedClassifierCV: A CalibratedClassifierCV containing necessary information 
         about the classifier.
    """
    clfSVM = LinearSVC()
    clfSVM = CalibratedClassifierCV(clfSVM)
    clfSVM.fit(trainData, trainLabels)
    
    return clfSVM


def setupRF(trainData, trainLabels):
    """
    Setup and train a classifier using RF.

    Args:
        trainData(numpy.ndarray): 2D array containing training data.
        trainLabels(list): List containing training labels.

    Returns:
         RandomForestClassifier: A RandomForestClassifier containing necessary information
         about the classifier.
    """
    
    clfRF = RandomForestClassifier(max_depth=6, random_state=0)
    clfRF.fit(trainData, trainLabels)
    
    return clfRF


def setupQDA(trainData, trainLabels):
    """
    Setup and train a classifier using QDA.
    
    Args:
        trainData(numpy.ndarray): 2D array containing training data.
        trainLabels(list): List containing training labels.

    Returns:
         QuadraticDiscriminantAnalysis: A QuadraticDiscimminantAnalysis containing necessary
         information about the classifier.
    """
    
    clfQDA = QuadraticDiscriminantAnalysis()
    clfQDA.fit(trainData, trainLabels)
   
    return clfQDA


def setupMPL(trainData, trainLabels):
    """
    Setup and train a classifier using MPL.

    Args:
        trainData(numpy.ndarray): 2D array containing training data.
        trainLabels(list): List containing training labels.
    
    Returns:
         MPLClassifier: A MPLClassifier containing necessary information about the classifier.
    """
   
    clfMPL = MLPClassifier(solver="lbfgs", random_state=0)
    clfMPL.fit(trainData, trainLabels)
    
    return clfMPL


def evalClf(clf, feats, testData, expectedLabels):
    """
    Evaluate the classifier clf.

    Args:
        clf(Classifier): A classifier containing necessary information about the classifier.
        feats(dataPackage): DataPackage containing data for ML algorithms.
        testData(numpy.ndarray): "D array containing testing data.
        expectedLabels(list): List containing the expected labels.

    Returns:
         Float: Float representing the mean score.
    """
   
    if isinstance(clf, KNeighborsClassifier):
        clfName = "k-NN"
    elif isinstance(clf, (LinearSVC, CalibratedClassifierCV)):
        clfName = "SVM"
    elif isinstance(clf, RandomForestClassifier):
        clfName = "RF"
    elif isinstance(clf, QuadraticDiscriminantAnalysis):
        clfName = "QDA"
    elif isinstance(clf, MLPClassifier):
        clfName = "MLP"
    else:
        clfName = "Unknown classifier"

    print("\nResults for {}\n".format(clfName))

    predictedLabels = clf.predict(testData)

    acc = metrics.accuracy_score(expectedLabels, predictedLabels)
    print("Accuracy: {}\n".format(acc))

    report = metrics.classification_report(expectedLabels, predictedLabels)
    print("Classification report:\n{}".format(report))

    confMat = metrics.confusion_matrix(expectedLabels, predictedLabels)
    print("Confusion matrix:\n{}\n".format(confMat))

    kFold = 10
    scores = cross_validate(clf, feats.features, feats.labels,
                            cv=kFold)["test_score"]
    print("Cross-validation scores:\n{}\n".format(scores))

    meanScore = np.mean(scores)
    print("Mean score: {}\n".format(meanScore))


def saveClf(clf, fileName="clf.dump"):
    """
    Save the classifier clf to the file fileName.
    
    Args:
        clf(classifier): Classifier containing necessary information about the classifier.
        fileName(str): The path to a file in which the classifier information will be stored.
    """
    joblib.dump(clf, fileName)


def loadClf(fileName="clf.dump"):
    """
    Load and return a classifier from the file fileName.

    Args:
        fileName (str): The path to a file storing a classifier.

    Returns:
         The classifier stored in the file fileName.
    """
    return joblib.load(fileName)


def getImportantFeatIndices(clfRF, n=10):
    """
    Gets the indices of the n most
    important features

    Args:
        clfRF(RandomForestClassifier): RandomForestClassifier containing necessary information 
        about the classifier.
        n(int): Int representing how many indices to pick.

    Returns:
         List: List of the n most important indices.
    """
    
    f = clfRF.feature_importances_
    featuresImportance = sorted(range(len(f)), key=lambda x: f[x])[-n:]
   
    return featuresImportance

def predictEmo(clf, landmarks):
    """
    Predict an emotion based on the landmarks in landmarks using
    the classifier clf.

    Args:
        clf(classifier): Classifier containing necessary information about the classifier.
        landmarks(numpy.ndarray): 2D array containing the landmarks.

    Returns:
        numpy.ndarray: A 1D array containing the probabilities for the emotions.
    """
   
    landmarks = np.array([landmarks])
    features = calcFeats(landmarks)
    probabilities = clf.predict_proba(features)[0]
    
    return probabilities


def colormap(nClasses):
    """
    Generate and return a Colormap instance and a Normalize instance.

    Args:
        nClasses(int): Int representing the number of classes.

    Returns:
         Tuple: Tuple representing the colormap and the boundary norm.
    """
    
    cmap = cm.jet
    # Extract all colors from the .jet map.
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # Create the new map.
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # Define the bins and normalize.
    bounds = np.linspace(0, nClasses, nClasses + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
   
    return (cmap, norm)


def visualize(feats, cmap, norm, transformer):
    """
    Visualize the data in feats.

    Args:
        feats(dataPackage): DataPackage containing data for ML algorithms.
        cmap(Colormap): A colormap.
        norm(BoundaryNorm): A boundary norm.
        transformer(PCA) A PCA containing necessary information.
    """
    
    xTrans = transformer.fit_transform(feats.features)
    pyplot.scatter(xTrans[:, 0], xTrans[:, 1], c=feats.labels, s=80,
                   cmap=cmap, norm=norm)
    pyplot.show()


def main(argv):
    """
    Predict an emotion from a set of 3D landmarks.

    This is done with machine learning algorithms trained on data
    from the directory specified by an item in argv starting with "dir=",
    or from the directory BosphorusDB_lm3 if not specified.
    """
    # Remove these two declarations and put the values directly in varDict?
    dataDir = "BosphorusDB_lm3"
    enableVisualization = False

    varDict = {
        "dir": dataDir,
        "vis": enableVisualization,
    }

    funcDict = {
        "dir": argvParser.dictAssign,
        "vis": argvParser.dictToggle,
    }

    argvParser.parseArgv(argv, varDict, funcDict)

    # Assign to these variables for clarity or use varDict directly?
    dataDir = varDict["dir"]
    enableVisualization = varDict["vis"]

    feats = DataPackage(dataDir, simple=False)

    (trainData, trainLabels, trainImageNames,
     testData, expectedLabels, testImageNames) = splitData(feats, 0.7)

    if (enableVisualization):
        cmap, norm = colormap(feats.nClasses)
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, init='pca', random_state=0)

        visualize(feats, cmap, norm, pca)
        visualize(feats, cmap, norm, tsne)

    clfKNN = setupKNN(trainData, trainLabels)
    clfSVM = setupSVM(trainData, trainLabels)
    clfRF = setupRF(trainData, trainLabels)
    clfQDA = setupQDA(trainData, trainLabels)
    clfMPL = setupMPL(trainData, trainLabels)
    
    saveClf(clfKNN, "clfKNN.dump")
    saveClf(clfSVM, "clfSVM.dump")
    saveClf(clfRF, "clfRF.dump")
    saveClf(clfQDA, "clfQDA.dump")
    saveClf(clfMPL, "clfMPL.dump")

    evalClf(clfKNN, feats, testData, expectedLabels)
    evalClf(clfSVM, feats, testData, expectedLabels)
    evalClf(clfRF, feats, testData, expectedLabels)
    evalClf(clfQDA, feats, testData, expectedLabels)
    evalClf(clfMPL, feats, testData, expectedLabels)


if __name__ == "__main__":
    main(sys.argv)
