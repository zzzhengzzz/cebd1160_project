#!/usr/bin/env python
#pip install opencv-python

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.datasets import load_digits
digits = datasets.load_digits()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
import scipy
import cv2
from fractions import Fraction
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve


def image2Digit(image):
    #adjust size to 8*8
    im_resized = scipy.misc.imresize(image,(8,8))
    #RGB(3D) adjust to gray 1D
    im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    #train 0-16, resolutaion - 16/255
    im_hex = Fraction(16,255) * im_gray
    #reverse to black background and white number
    im_reverse = 16 - im_hex
    return im_reverse.astype(np.int)

#split training sets
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)
#create
clf = LogisticRegression(penalty='l2')
#combine
clf.fit(Xtrain, ytrain)
#predict
ypred = clf.predict(Xtest)
#calculate the accuracy
accuracy = accuracy_score(ytest, ypred)
print("Recognition_accuracyLR:",accuracy)


from sklearn import svm
from sklearn.svm import SVC

clf = svm.SVC(gamma=0.001, C=100.)
x,y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)
#print('Prediction:', clf.predict(digits.data[-1:]))
#plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()
x = digits.data[:-1]
y = digits.target[:-1]

#split data into test and training set
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.30, random_state=42,stratify=y)

clf.fit(train_x, train_y)
y_predict = clf.predict(test_x)
accuracy_s = accuracy_score(test_y,y_predict)
print("Recognition_accuracySVC:",accuracy_s)

###https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/###
# Create feature matrix and target vector
X, y = digits.data, digits.target

# Create range of values for parameter
param_range = np.arange(1, 250, 2)

# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(RandomForestClassifier(),
                                             X,
                                             y,
                                             param_name="n_estimators",
                                             param_range=param_range,
                                             cv=3,
                                             scoring="accuracy",
                                             n_jobs=-1)


# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("Validation Curve With Random Forest")
plt.show()
