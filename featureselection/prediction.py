import math
import data
import selection

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import preprocessing

import matplotlib.pyplot as plt



model_effort_rf_RFE = [
    'Input.reward',
    'input.type.radio',
    'textarea.count',
    "img.count",
    'Type',
    'Words',
    'Subclauses',
    '"100% of windows have density equal to or less"',

    'AvgWordLength',
    'Input.timeAlloted',
    'The average size of connected regions',
    'avgUniqueStems',
    'characters',
    'threshold',
    'p',
    'Input.hitsAvailable',
    'span',
    'histogram_0',
    'windows greater than threshold',
    'input.count',
    'histogram_4',
    'fileSize',
    'Sentences',
    'numUniqueStems',
    'histogram_8',
    'histogram_5',
    'div',
]

model_effort_rf = [
    # mturk features
    "Input.reward",

    # html features
    "input.type.radio",
    "textarea.count",
    "img.count",

    # NLP features
    "Words",
    "Subclauses",
    "Type",

    # image features
    '"100% of windows have density equal to or less"',
]

model_wouldtake_svm = [
    # most important
    'Answer.Q_FREQUENCY',
    'Answer.Q_Intresting',

    'Answer.Q_COMPLEX',
    'Answer.Q_DIFFICULT',
]

def testModel(X, Y, W, headings, model, predict, isDiscrete, SD=None):
    # normalize features column wise
    for i in range(0, X.shape[1]):
        X[:, i:i + 1] = preprocessing.normalize(X[:, i:i + 1], axis=0)

    # filter out features based on prediction model
    X_, headings_ = selection.selectFeatures(X, headings, model)

    meanMeanError = 0
    numMeanInside = 0
    numMeanWrong = 0
    numIterations = 20
    meanStd = 0

    meanCoefs = numpy.zeros((len(headings_)), dtype=float)
    for i in range(0, numIterations):
        # use indicies to split matricies randomly but in order
        split = 0.6
        X1, X2, indicies = data.splitMatrixRandomly(X_, split)
        Y1, Y2, indicies = data.splitMatrixRandomly(Y, split, indicies)
        if SD is not None:
            sd1, sd2, indicies = data.splitMatrixRandomly(SD, split, indicies)
        if W is not None:
            w1, w2, indicies = data.splitMatrixRandomly(W, split, indicies)

        # flatten data
        Y1 = Y1.flatten()
        Y2 = Y2.flatten()
        if W is not None:
            w1 = w1.flatten()
            w2 = w2.flatten()
        if SD is not None:
            sd1 = sd1.flatten()
            sd2 = sd2.flatten()

        if W is not None:
            predTrain, predTest, clf = predict(X1, X2, Y1, Y2, w1)
        else:
            predTrain, predTest, clf = predict(X1, X2, Y1, Y2, None)

        diff = 0
        numInside = 0
        numWrong = 0
        for p, val in enumerate(predTrain):
            if SD is not None:
                ci = 1.96 * (sd1[p] / 3.872983)
                if abs(val - Y1[p]) < ci:
                    numInside += 1
            if val != Y1[p]:
                numWrong += 1
            diff += abs(val - Y1[p])
        for p, val in enumerate(predTest):
            if SD is not None:
                ci = 1.96 * (sd2[p] / 3.872983)
                if abs(val - Y2[p]) < ci:
                    numInside += 1
            if val != Y2[p]:
                numWrong += 1
            diff += abs(val - Y2[p])


        std1 = numpy.sum(numpy.power(predTrain - Y1, 2))
        std2 = numpy.sum(numpy.power(predTest - Y2, 2))
        std = math.sqrt((std1 + std2) / (Y1.shape[0] + Y2.shape[0]))
        meanStd += std

        meanMeanError += diff / (Y1.shape[0] + Y2.shape[0])
        numMeanInside += numInside / (Y1.shape[0] + Y2.shape[0])
        numMeanWrong += numWrong / (Y1.shape[0] + Y2.shape[0])

        try:
            for c, coef in enumerate(clf.coef_[0]):
                meanCoefs[c] += coef
        except:
            True
        try:
            for c, coef in enumerate(clf.feature_importances_):
                meanCoefs[c] += coef
        except:
            True


        if i == numIterations - 1:
            print('\n--- Feature Rankings ---')
            for i in range(0, meanCoefs.shape[0]):
                print(headings_[i], meanCoefs[i] / numIterations)
            print('')


            if not isDiscrete:
                print('error: ', diff / (Y1.shape[0] + Y2.shape[0]))
                print('std:', std)
                print("mean error:", meanMeanError / numIterations, "@iter_" + str(numIterations))
                print("mean std:", meanStd / numIterations, "@iter_" + str(numIterations))
                print("inside 95%:", numMeanInside / numIterations, "@iter_" + str(numIterations))
            else:
                print('error rate:', numMeanWrong / numIterations, "@iter_" + str(numIterations))
            plt.xlim(0, max(Y2))
            plt.ylim(0, max(Y2))
            plt.plot(Y2, Y2, 'bo')
            plt.plot(Y1, predTrain, 'ro')
            plt.plot(Y2, predTest, 'ro')
            if not isDiscrete:
                plt.show()



# -------------------- EFFORT ------------------------------

def predictEffort(X1, X2, Y1, Y2, W):
    rf = RandomForestRegressor()
    if W is None:
        rf.fit(X1, Y1)
    else:
        rf.fit(X1, Y1, W)
    predTrain = rf.predict(X1)
    predTest = rf.predict(X2)
    return predTrain, predTest, rf

def testModelEffort(model):
    X, Y, W, headings = data.loadFeaturesAndLabels(data.DV_Type.EFFORT_TRIMMED)
    # calculate weights from standard deviation
    weights = numpy.zeros((W.shape[0], 1), dtype=float)
    for i, sd in enumerate(W):
        weights[i, 0] = 1 - (sd / 30)

    if model is None:
        model = headings

    testModel(X, Y, weights, headings, model, predictEffort, False, W)




# ---------------- WOULD TAKE ---------------------

def predictWouldTakeRF(X1, X2, Y1, Y2, W):
    clf = RandomForestRegressor()
    clf.fit(X1, Y1)
    predTrain = clf.predict(X1)
    predTest = clf.predict(X2)
    return predTrain, predTest, clf

def predictWouldTakeSVM(X1, X2, Y1, Y2, W):
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X1, Y1)
    predTrain = clf.predict(X1)
    predTest = clf.predict(X2)
    return predTrain, predTest, clf

def predictWouldTakeLR(X1, X2, Y1, Y2, W):
    clf = LogisticRegression()
    clf.fit(X1, Y1)
    predTrain = clf.predict(X1)
    predTest = clf.predict(X2)
    return predTrain, predTest, clf

def testWouldTake(predictor):
    X, Y, W, headings = data.loadFeaturesAndLabels(data.DV_Type.WOULD_TAKE)

    # remove all where would take ambiguous
    print('removing ambiguous would take ratings')
    threshold = 8
    deleted = 0
    yes = 0
    no = 0
    for i, row in enumerate(Y):
        index = i - deleted
        if row[0] >= threshold and row[0] < 20 - threshold:
            X = numpy.delete(X, index, 0)
            Y = numpy.delete(Y, index, 0)
            deleted += 1
        elif row[0] < threshold:
            Y[index, 0] = 0
            no += 1
        elif row[0] >= 20 - threshold:
            Y[index, 0] = 1
            yes += 1

    print('removed - ambiguous:', deleted)
    print('remaining:', X.shape[0], 'yes:', yes / X.shape[0], 'no:', no / X.shape[0])

    #testModel(X, Y, None, headings, model_wouldtake_svm, predictWouldTakeSVM, True)
    testModel(X, Y, None, headings, model_wouldtake_svm, predictor, True, None)




#print('Predict: RF -> WOULD TAKE:')
#testWouldTake(predictWouldTakeRF)
#print('--------- ### ---------')
#print('')
#print('Predict: SVM -> WOULD TAKE:')
#testWouldTake(predictWouldTakeSVM)
#print('--------- ### ---------')
#print('')
print('Predict: Logistic Regression -> WOULD TAKE:')
testWouldTake(predictWouldTakeLR)
print('--------- ### ---------')
print('')
#print('Predict: Random Forest ALL -> EFFORT:')
#testModelEffort(None)
#print('--------- ### ---------')
#print('')
#print('Predict: Random Forest RFE -> EFFORT:')
#testModelEffort(model_effort_rf_RFE)
#print('--------- ### ---------')
#print('')
#print('Predict: Random Forest RFE refined -> EFFORT:')
#testModelEffort(model_effort_rf)
#print('--------- ### ---------')

#selection.RFE_effortRandomForest(20, True)