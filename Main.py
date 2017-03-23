from Classify import Classify
import time


if __name__ == "__main__":
    start = time.clock()
    classify = Classify()
    trainingPath = "C:\\Users\\ace\\Desktop\\newData\\training"
    testPath = "C:\\Users\\ace\\Desktop\\newData\\testing"

    featureExtraType = "surf"
    result = classify.algoSVM(trainingPath, testPath, featureExtraType)

    end  = time.clock()
    print "run time: %f s" % (end - start)

    print result
    print "Taux global:%f" % (result.trace() / result.sum())


