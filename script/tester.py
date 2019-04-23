from statisticalClassifer import statisticalClassifier
from featureExtractor import featureExtractor
from dataLoader import dataLoader


obj1 = dataLoader()
obj1.loadData()

obj2 = featureExtractor()
obj2.getFeatures()

obj3 = statisticalClassifier()
obj3.runClassifiers()
