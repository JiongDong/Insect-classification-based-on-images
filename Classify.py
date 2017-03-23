from LoadImage import LoadImage
from FeatureExtractor import FeatureExtractor
from BOW import BOW
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import cv2
import numpy as np
import os


class Classify:
    def __init__(self):
        pass


    def algoSVM(self,trainingPath, testPath, featureExtraType):
        '''
        Using Support Vector Machine algorithm to classify insect images
        :param trainingPath: the path of training images
        :param testPath: he path of testing images
        :param featureExtraType: the feature type : sift or surf
        :return:
        '''
        loadImage = LoadImage()
        featureExtra = FeatureExtractor()
        bow = BOW()

        #get the species,the name of all insects, the path of all insect images
        insectSpecies, names, trainingPaths = loadImage.loadTrainImage(trainingPath)
        print insectSpecies
        print "Le bombre d'espece :", len(insectSpecies)
        dictionarySize = len(insectSpecies)

        insect = {}
        num = 1
        for name in insectSpecies:
            insect[name] = num
            num += 1

        #get the descriptors of all training images
        descriptors = bow.getDescriptors(trainingPaths, featureExtraType)
        #get Bag of Words dictionary
        bowDictionary = bow.getBowDictionary(dictionarySize, descriptors, featureExtraType)
        print "bow dictionary"

        #train data
        trainDesc = []
        #train response
        trainLabels = []
        i = 0

        #initialize train datas and train responses
        for p in trainingPaths:
            trainDesc.extend(featureExtra.getSingleFeature(p, bowDictionary, featureExtraType))
            trainLabels.append(insect[names[i]])
            i = i + 1

        svm = cv2.SVM()
        #training
        svm.train(np.array(trainDesc), np.array(trainLabels))

        testInsectNames = os.listdir(testPath)
        # Initialize a zero matrix to save the classification results
        result = np.zeros((dictionarySize, dictionarySize))
        print "result zero"

        count = 0
        #classify all the test immages
        for test in testInsectNames:
            testingImage = os.listdir(testPath + "\\" + test)
            for p in testingImage:
                #get feature from a test image
                feature = featureExtra.getSingleFeature(testPath + "\\" + test + "\\" + p, bowDictionary, featureExtraType)
                #predict
                p = svm.predict(feature)
                #save the result in the result matrix
                result[count, p - 1] += 1

            count += 1

        return result

    def algoANN(self, trainingPath, testPath, featureExtraType, epochs):
        '''
        Using Artificial Neural Network algorithm to classify insect images
        :param trainingPath: the path of training images
        :param testPath:  the path of testing images
        :param featureExtraType:  the feature type:sift or surf
        :param epochs: the numbre of training for neural network
        :return: the classification results
        '''
        loadImage = LoadImage()
        featureExtra = FeatureExtractor()
        bow = BOW()

        # get the species,the name of all insects, the path of all insect images
        insectNames, names, trainingPaths = loadImage.loadTrainImage(trainingPath)
        insectSpecies = len(insectNames)
        print "insect species:", insectSpecies

        #get all descriptos of training images
        trainDescriptors = bow.getDescriptors(trainingPaths, featureExtraType)
        #get the BoW dictionary of trianing images
        trainBowDictionary = bow.getBowDictionary(insectSpecies, trainDescriptors, featureExtraType)

        #initialize a Neural Network
        net = buildNetwork(insectSpecies, 100, 100, insectSpecies)
        #initialize a data set
        ds = SupervisedDataSet(insectSpecies, insectSpecies)
        species = 0
        #add all datas in data set
        for p in insectNames:
            trainingPaths = os.listdir(trainingPath + "\\" + p)
            for j in trainingPaths:
                #add data
                ds.addSample(featureExtra.getSingleFeature(trainingPath + "\\" + p + "\\" + j,
                                                           trainBowDictionary, featureExtraType)[0], (species,))
            species += 1

        #initialize a trainer
        trainer = BackpropTrainer(net, ds, learningrate=0.01, momentum=0.1, weightdecay=0.01)
        #training
        for i in range(1, epochs):
            traError = trainer.train()
            print 'after %d epochs,train error:%f' % (i, traError)

        testInsectNames, testNames, testingPaths = loadImage.loadTrainImage(testPath)
        testDescriptors = bow.getDescriptors(testingPaths, featureExtraType)
        testBowDictionary = bow.getBowDictionary(insectSpecies, testDescriptors, featureExtraType)
        # Initializes a zero matrix to save the classification results
        result = np.zeros((insectSpecies, insectSpecies))

        count = 0
        # classify all the test immages
        for m in testInsectNames:
            testPaths = os.listdir(testPath + "\\" + m)
            for n in testPaths:
                test = net.activate(featureExtra.getSingleFeature(testPath + "\\" + m + "\\" + n,
                                                                  testBowDictionary, featureExtraType)[0])
                target = map(lambda x: (x), test)  # numpy.array to list
                result[count, target.index(max(target))] += 1
            count += 1

        return result

