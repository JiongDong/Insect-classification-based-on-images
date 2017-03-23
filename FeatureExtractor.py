import cv2


class FeatureExtractor:
    def __init__(self):
        pass

    def getSiftFeature(self, image):
        '''
        get the sift features from a insect image
        :param image:
        :return:the descriptor of the image
        '''
        sift = cv2.SIFT()
        imgBlur = cv2.GaussianBlur(image, (5, 5), 0)  # Remove noise
        gray = cv2.cvtColor(imgBlur, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        kp, dsc = sift.detectAndCompute(gray, None)  # get keypoint and descriptor
        return dsc

    def getSurfFeature(self, image):
        '''
        get the surf features from a insect image
        :param image:
        :return: the descriptor of the image
        '''
        surf = cv2.SURF()
        imgBlur = cv2.GaussianBlur(image, (5, 5), 0)  # Remove noise
        gray = cv2.cvtColor(imgBlur, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        kp, dsc = surf.detectAndCompute(gray, None)  # get keypoint and descriptor
        return dsc


    def getSingleFeature(self, path, bowDictionary, featureExtraType):
        '''
        get an image's feature by using Bag of words dictionary
        :param path:image's path
        :param bowDictionary:the dictionary of Bag of words
        :param featureExtraType:the feature type:sift or surf
        :return:
        '''
        im = cv2.imread(path, 1)
        gray = cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if (featureExtraType.upper() == "SIFT"):
            return bowDictionary.compute(gray, cv2.SIFT().detect(gray))
        if (featureExtraType.upper() == "SURF"):
            return bowDictionary.compute(gray, cv2.SURF().detect(gray))