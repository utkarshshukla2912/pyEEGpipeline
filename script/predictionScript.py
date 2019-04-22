from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import itertools
import warnings
import pickle
import json



class prediction

    def __init__():
        with open('config.json') as f:
            config = json.load(f)


def strToFeatures(string):
    string = string[1:-1]
    return list(map(float,string.split(',')))

def plot_confusion_matrix(cm, classes,normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

file = pd.read_csv('../objects/dataFeatures/channel_split_features.csv',
                   header = None,error_bad_lines=False)
columns = ['subject_id','experiment_identifier','channel_name',
           'split_number','feature_extracted']
file.columns = columns
label_string = file['experiment_identifier'].values
feature_string = file['feature_extracted'].values
features = list(map(strToFeatures,feature_string))
labels = LabelEncoder().fit_transform(label_string)

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                    test_size=0.33, random_state=42)

names = ["LinearSVM","KernalSVM","DecisionTree","LogisticRegression",
         "MLPClassifier","KNeighboursClassifier","GradientBoost",
         "RandomForest","AdaBoost","GaussianNB","QuadraticDescriminant"
        ]

classifiers = [
               SVC(kernel="linear", C=0.025),SVC(kernel='sigmoid',gamma=1, C=0.01),
               DecisionTreeClassifier(max_depth=100),LogisticRegression(),MLPClassifier(),
               KNeighborsClassifier(),GradientBoostingClassifier(),
               RandomForestClassifier(max_depth=50,n_estimators = 20),
               AdaBoostClassifier(),GaussianNB(),QuadraticDiscriminantAnalysis()
              ]



for name, clf in zip(names, classifiers):
    try:
        clf.fit(X_train,y_train)
        print ("Calculating for model: ",name)
        #filename = name+'_{}.sav'.format(config['model_to_use'])
        #pickle.dump(clf, open(object_path+model_path+filename, 'wb'))
        print (name+' Score:',clf.score(X_test,y_test))
        prediction = clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test,prediction)
        np.set_printoptions(precision=2)
        print (name+" Recall metric in the testing dataset: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1])))
        class_names = [0,1]
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, title=name)
        print (name+" Precision: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1])))

        recall=np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1]))
        print (name+" Recall: ", recall)
        precision=np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))
        f1score=2*np.true_divide(precision*recall,(precision+recall))
        print (name+" F1 Score: ", f1score,"\n\n")
        plt.show()
    except Exception as e:
        print(e)
        pass




#done
