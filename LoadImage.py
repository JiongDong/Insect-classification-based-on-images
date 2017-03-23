import os


class LoadImage:
    def __init__(self):
        pass

    def loadTrainImage(self,path):
        '''
        load all train images from the path
        :param path: the images' path
        :return: the type and the name of all insects, the path of all insect images
        '''

        # get all species' names
        insectSpecies = os.listdir(path)

        trainingPaths = []
        names = []
        # get full list of all training images
        for p in insectSpecies:
            paths = os.listdir(path + "\\" + p)
            for j in paths:
                trainingPaths.append(path + "\\" + p + "\\" + j)
                names.append(p)

        return insectSpecies,names,trainingPaths

