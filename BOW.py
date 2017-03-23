from FeatureExtractor import FeatureExtractor
import cv2


class BOW:
    def __init__(self):
        pass

    def getBowDictionary(self,dictionarySize, descriptors, featureExtraType):
        '''
        get the dictionary of Bog of words
        :param dictionarySize: the size of dictionary
        :param descriptors: the all images' descriptors
        :param featureExtraType: the feature type: sift or surf
        :return: the dictionary of Bag of words
        '''
        BOW = cv2.BOWKMeansTrainer(dictionarySize)
        for dsc in descriptors:
            BOW.add(dsc)

        # dictionary created
        dictionary = BOW.cluster()

        if (featureExtraType.upper() == "SIFT"):
            extra = cv2.DescriptorExtractor_create("SIFT")
        if (featureExtraType.upper() == "SURF"):
            extra = cv2.DescriptorExtractor_create("SURF")
        bowDictionary = cv2.BOWImgDescriptorExtractor(extra, cv2.BFMatcher(cv2.NORM_L2))
        bowDictionary.setVocabulary(dictionary)
        return bowDictionary

    def getDescriptors(self, path, featureExtraType):
        '''
        get all descriptors from images in the path
        :param path: the image's path
        :param featureExtraType: the feature type:sift or surf
        :return: all images' descriptors
        '''
        featureExtra = FeatureExtractor()
        descriptors = []
        for p in path:
            image = cv2.imread(p)
            if (featureExtraType.upper() == "SIFT"):
                dsc = featureExtra.getSiftFeature(image)
            if (featureExtraType.upper() == "SURF"):
                dsc = featureExtra.getSurfFeature(image)
            descriptors.append(dsc)

        return descriptors