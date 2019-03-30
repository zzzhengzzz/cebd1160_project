# cebd1160_project

Instructions and template for final projects.

| Name | Date |
|:-------|:---------------|
|Zheng Zheng | 2019/03/29|

-----

### Resources
This repository includes the following:

- Python script for your analysis: `digits.py`
- Results figure/saved file: `figures/`
- Dockerfile for your experiment: `Dockerfile`
- runtime-instructions in a file named `RUNME.md`


-----

## Research Question

What is the accuracy of recognizing digits? 

### Abstract

The data set contains hand-written digits images, with 10 classes (0-9). Using Logistic Regression and SVM to recognize the data set, there is the difference in accuracy. Based on the performance of the both methods, we found that SVM has slightly higher accuracy.

### Introduction

The data set contains images of hand-written digits: 10 classes where each class refers to a digit.
(https://scikit-learn.org/stable/datasets/index.html#digits-dataset)


### Methods

The method for modelling the data were Logistic Regression and Support Vector Classification(SVC) built into scikit-learn.
Pseudocode: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

Logistic Regression has 0.946666666667 accuracy
SVM has 0.9888682745825603 accuracy


### Results

![alt text](https://github.com/mikeditri/Final_Project/blob/master/figures/RFC_Accuracy.png)

Choosing learning algorithms and hyperparameters can reduce bias and variance to as low as possible, using more training data can help to reduce variance. Pseudocode: ( https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/)

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
All of the links

-------
