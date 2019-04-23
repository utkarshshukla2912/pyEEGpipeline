from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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




class statisticalClassifier:

	def __init__ (self):

		with open('support/config.json') as f:
			config = json.load(f)

		# preprocessing techniques (PCA, StandardScaler)
		self.classifiersToTrain = config['classifiersToTrain']
		self.featureExtracted = config['featureExtracted']
		self.classToClassify = config['classToClassify']
		self.channelsToUse = config['channelsToUse']
		self.dataFeatures = config['dataFeatures']
		self.applyScaling = config['applyScaling']
		self.objectPath = config['objectPath']
		self.applyPCA = config["applyPCA"]

		self.data = pd.read_csv(self.objectPath + self.dataFeatures + self.featureExtracted)
		self.data = self.data.rename(columns={'bandPower_0': 'delta', 'bandPower_1': 'theta',
						'bandPower_2':'alpha','bandPower_3':'beta',
						'bandPower_4': 'delta_ratio', 'bandPower_5': 'theta_ratio',
						'bandPower_6':'alpha_ratio','bandPower_7':'beta_ratio',
					   })

		self.classifierName = ["DecisionTree", "LogisticRegression", "MLPClassifier",
				 "KNeighboursClassifier", "GradientBoost", "RandomForest",
				 "LinearSVM","KernalSVM-RBF","KernalSVM-Poly"]

		self.classifierFunc = [
								DecisionTreeClassifier(), LogisticRegression(),
								MLPClassifier(),KNeighborsClassifier(),GradientBoostingClassifier(),
								RandomForestClassifier(), LinearSVC(), SVC(),
								SVC(kernel = 'poly')
								]

		self.classifierDict = dict(zip(self.classifierName,self.classifierFunc))
		self.classifierDict = self.filterClassifiers(self.classifierDict,
													self.classifiersToTrain)


	def filterClassifiers(self,dictionary, keys):
		"""Filters a dict by only including certain keys."""
		key_set = set(keys) & set(dictionary.keys())
		return {key: dictionary[key] for key in key_set}


	def prepareData(self):

		if self.classToClassify != 'all':
			self.data = self.data.loc[self.data['experiment_identifier'].isin(self.classToClassify)]
		if self.channelsToUse != 'all':
			self.data = self.data.loc[self.data['channel_name'].isin(self.channelsToUse)]


		self.features = np.array(self.data[self.data.columns[4:]].values,
								 dtype=float)
		if self.applyScaling == "y":
			self.features = StandardScaler().fit_transform(self.features)
		if self.applyPCA[0] == "y":
			self.features = PCA(n_components=int(self.applyPCA[1])).fit_transform(self.features)

		label_string = self.data['experiment_identifier'].values
		self.labels = LabelEncoder().fit_transform(label_string)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
					self.features, self.labels, test_size=0.33, random_state=42)


	def plot_confusion_matrix(self,cm, classes,normalize=True,
							  title='Confusion matrix',cmap=plt.cm.Blues):

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=0)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		cm = np.around(cm,decimals=2)
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')


	def runClassifiers(self):

		self.prepareData()
		for name, clf in zip(list(self.classifierDict.keys()),list(self.classifierDict.values())):
			print('Classifying Using:',name)
			clf.fit(self.X_train, self.y_train)
			score = clf.score(self.X_test, self.y_test)
			y_pred = clf.predict(self.X_test)
			prediction = clf.predict(self.X_test)

			cnf_matrix = confusion_matrix(self.y_test,y_pred)
			np.set_printoptions(precision=2)
			class_names = ['negative', 'positive']
			recall=np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1]))
			precision=np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))
			f1score=2*np.true_divide(precision*recall,(precision+recall))

			print ("\n"+name+" Accuracy: ",accuracy_score(self.y_test,y_pred))
			print (name+" Recall: ", recall)
			print ("\n"+name+" Precision: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1])))
			print (name+" F1 Score: ", f1score)

			y_pred_proba = clf.predict_proba(self.X_test)
			y_pred_probability = clf.predict_proba(self.X_test)
			y_pred_proba = y_pred_proba[:,1]
			self.plot_confusion_matrix(cnf_matrix,class_names)
			plt.show()








#done
