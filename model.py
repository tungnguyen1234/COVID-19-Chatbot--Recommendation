import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, Lasso
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot



sdg_params = dict(alpha=1e-5, penalty="l2", loss="log")
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8, token_pattern=r"(?u)\b\w+\b", stop_words=None, analyzer='word')

def data_processing(main_path, file1, file2):
    r1 = open(main_path + file1, 'r')
    r2 = open(main_path + file2, 'r')
    line1 = r1.readlines()
    line2 = r2.readlines()
    random.shuffle(line1)
    random.shuffle(line2)
    X = []
    y = []
    for l in line1:
        l = l.strip()
        strs = l.split('\t')
        if strs != ['']:
            X.append(strs[0])
            y.append(strs[1])

    for l in line2:
        l = l.strip()
        strs = l.split('\t')
        if strs != ['']:
            X.append(strs[0])
            y.append(strs[1])
    y = np.array(y)
    return X,y

pipeline1 = Pipeline(
[
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", SGDClassifier(**sdg_params)),
]
)

pipeline2 = Pipeline(
[
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", MultinomialNB()),
]
)

pipeline3 = Pipeline(
[
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", LogisticRegression()),
]
)


pipeline4 = Pipeline(
[
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", Perceptron(tol=1e-3, random_state=0)),
]
)

# Plotting ROC curves
def run_auc(clf, X_test, y_test):
    integer = lambda t: int(t)
    y_test = np.vectorize(integer)(y_test)
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    lr_probs = clf.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Method in consideration: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    plt.figure(3)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    print("-" * 10)

def model(clf, X_train, X_test, y_train, y_test):
    print("Number of training samples:", len(X_train))
    print(type(X_train[0]))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_check = clf.predict(X_train)
    print(
        "Micro-averaged accuracy score on test set: %0.3f"
        % accuracy_score(y_test, y_pred)
    )
    print(
        "Micro-averaged accuracy score on training set: %0.3f"
        % accuracy_score(y_train, y_check)
    )
    plt.figure(1)
    plot_confusion_matrix(clf, X_test, y_test)
    plt.figure(2)
    plot_confusion_matrix(clf, X_train, y_train)
    plt.show()
    run_auc(clf, X_test, y_test)

    # plt.scatter(y_test, y_pred)
    # plt.savefig('confusion_matrix_new_1.png')
    
   
    
    
def model_cv(clf, X, y):
    print("Number of samples:", len(X))
    # cv=KFold(n_splits=5)
    cv = LeaveOneOut()
    score = cross_val_score(clf,X,y, cv=cv)
    print("Average Cross Validation score :{}".format(score.mean()))


