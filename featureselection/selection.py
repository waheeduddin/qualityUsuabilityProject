import math
import data

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE
from minepy import MINE
from sklearn import svm
from sklearn import preprocessing

import matplotlib.pyplot as plt

def selectFeatures(X, headings, featureNames):
    # get column indicies of selected features
    goodfeatures = []
    for name in featureNames:
        for i, h in enumerate(headings):
            if h == name:
                goodfeatures.append(i)
                break

    selected = numpy.zeros((X.shape[0], len(goodfeatures)), dtype=float)
    for i, col in enumerate(goodfeatures):
        selected[:, i:i + 1] = X[:, col:col + 1]

    return selected, featureNames



def pearson(features, labels):
    labels = labels.flatten()

    coeff = []
    for i in range(0, features.shape[1]):
        f = features[:, i].flatten()
        coeff.append(numpy.corrcoef(f, labels)[0, 1])

    return coeff

def logisticRegression(features, labels):
    labels = labels.flatten()
    for i in range(0, labels.shape[0]):
        labels[i] = math.floor(labels[i])

    model = LogisticRegression()
    model = model.fit(features, labels)

    return model.coef_

def linearRegression(features, labels, regularization=0):
    if regularization == 0:
        lr = LinearRegression()
        lr.fit(features, labels)
        return lr.coef_

    elif regularization == 1:
        lasso = Lasso()
        lasso.fit(features, labels)
        return lasso.coef_

    elif regularization == 2:
        ridge = Ridge()
        ridge.fit(features, labels)
        return ridge.coef_

def MIC(features, labels):
    mine = MINE()
    mic_scores = []

    labels = labels.flatten()
    for i in range(features.shape[1]):
        mine.compute_score(features[:, i], labels)
        m = mine.mic()
        mic_scores.append(m)

    return mic_scores

def stability(features, labels):
    labels = labels.flatten()
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(features, labels)
    return rlasso.scores_

def randomForest(features, labels):
    labels = labels.flatten()
    rf = RandomForestRegressor()
    rf.fit(features, labels)
    return rf.feature_importances_

# DODGY WAY TO DISCRETIZE LABELS - BAD DONT USE IT
# ONLY REMAINS HERE AS AN EXAMPLE FOR SVM IMPLEMENTATION
def SVM(features, labels):
    # flatten and discretize labels
    labels = labels.flatten()
    for i in range(0, labels.shape[0]):
        labels[i] = math.floor(labels[i])

    if False:
        # create close to linspace discrete space
        discrete = numpy.linspace(min(labels), max(labels), math.floor(labels.shape[0] * 1))
        for i in range(0, discrete.shape[0]):
            discrete[i] = math.floor(discrete[i])

        # reduce num labels by using discrete linspace as label estimator
        for i in range(0, labels.shape[0]):
            closest = 0
            for d in range(0, discrete.shape[0]):
                if abs(labels[i] - discrete[d]) < abs(labels[i] - closest):
                    closest = discrete[d]
            labels[i] = closest

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(features, labels)

    print(clf.coef_)
    return clf.coef_


def calcFeatureTable(dv_type):
    features, labels, weights, headings = data.loadFeaturesAndLabels(dv_type)

    # normalize features
    for i in range(0, features.shape[1]):
        features[:, i:i + 1] = preprocessing.normalize(features[:, i:i + 1], axis=0)

    # add feature names to table
    table = [['Features']]
    for heading in headings:
        table.append([heading])

    methods = [
        ['Lin. Cor.', lambda X, Y: pearson(X, Y)],
        ['Lin. Reg.', lambda X, Y: linearRegression(X, Y, 0)],
        ['Lasso', lambda X, Y: linearRegression(X, Y, 1)],
        ['Ridge', lambda X, Y: linearRegression(X, Y, 2)],
        ['MIC', lambda X, Y: MIC(X, Y)],
        ['Stability', lambda X, Y: stability(X, Y)],
        ['Random Forest', lambda X, Y: randomForest(X, Y)]
    ]

    for method in methods:
        print('applying', method[0])
        coefs = method[1](features, labels)

        # format coefs: some methods output scores in nested row
        vals = coefs
        try:
            if coefs.shape[1] == len(features[0]):
                vals = coefs[0]
        except:
            True

        # append method to headings
        table[0].append(method[0])

        # append features scores to table
        for i, c in enumerate(vals):
            if math.isnan(c):
                c = 0
            table[1+i].append(math.floor(c * 1000) / 1000)

    return table


def calculateTables():
    for dv_type in data.DV_Type:
        if dv_type != data.DV_Type.HIT_TYPE and dv_type != data.DV_Type.WOULD_TAKE:
            table = calcFeatureTable(dv_type)
            fname = '../data/selection/P_' + str(dv_type).split('.')[1]
            data.writeTableHTML(fname + '.html', table, str(dv_type).split('.')[1])

def RFE_effortRandomForest(numFeatures, formated = False):
    X, Y, W, headings = data.loadFeaturesAndLabels(data.DV_Type.EFFORT_TRIMMED)
    Y = Y.flatten()

    # normalize features column wise
    for i in range(0, X.shape[1]):
        X[:, i:i + 1] = preprocessing.normalize(X[:, i:i + 1], axis=0)

    clf = RandomForestRegressor()
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(clf, n_features_to_select=10)
    rfe.fit(X, Y)


    for i, val in enumerate(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), headings))):
        if formated:
            if i % 10 == 0:
                print(i)
            print("    '" + val[1] + "',")
        else:
            print(i, val)

def RandomFeatureRanking(use_zweights = False):
    dv = data.DV_Type.WORKLOAD_EFFORT
    X, Y, W, headings = data.loadFeaturesAndLabels(dv)

    # normalize features column wise
    for i in range(0, X.shape[1]):
        X[:, i:i + 1] = preprocessing.normalize(X[:, i:i + 1], axis=0)

    # calculate weights from standard deviation
    weights = numpy.zeros((W.shape[0], 1), dtype=float)
    for i, sd in enumerate(W):
        weights[i, 0] = 1 - (sd / 30)


    randoms = numpy.zeros((1000, 4), dtype=float)

    for r in range(0, randoms.shape[0]):
        print("progress:", r / randoms.shape[0])
        test = numpy.arange(X.shape[1])
        numpy.random.shuffle(test)
        test = test[:randoms.shape[1]-1]
        selected = []
        for t, i in enumerate(test):
            selected.append(headings[i])

        # filter out features based on prediction model
        X_, headings_ = selectFeatures(X, headings, selected)


        numIterations = 10
        meanMeanError = 0
        for i in range(0, numIterations):
            # use indicies to split matricies randomly but in order
            split = 0.8
            X1, X2, indicies = data.splitMatrixRandomly(X_, split)
            Y1, Y2, indicies = data.splitMatrixRandomly(Y, split, indicies)
            w1, w2, indicies = data.splitMatrixRandomly(weights, split, indicies)

            # flatten data
            Y1 = Y1.flatten()
            Y2 = Y2.flatten()
            w1 = w1.flatten()
            w2 = w2.flatten()

            rf = RandomForestRegressor()
            rf.fit(X1, Y1)
            pred = rf.predict(X2)

            diff = 0
            for p, val in enumerate(pred):
                diff += abs(val - Y2[p])
            #print(diff / Y2.shape[0])


            meanMeanError += diff / Y2.shape[0]

            if False:
                plt.xlim(0, max(Y2))
                plt.ylim(0, max(Y2))
                plt.plot(Y2, Y2, 'bo')
                plt.plot(Y2, pred, 'ro')
                plt.ylabel(str(dv))
                plt.show()

        for i in range(0, randoms.shape[1]):
            randoms[r, 0] = meanMeanError / numIterations
            randoms[r, 1:] = test


    randoms = randoms[numpy.argsort(randoms[:, 0])]

    test = []
    for r, row in enumerate(randoms):
        if len(test) > 10:
            break
        for l, val in enumerate(row):
            if l > 0:
                test.append(val)

    # insert headings + avoid doubles
    selected = []
    for t, i in enumerate(test):
        h = headings[int(i)]
        try:
            selected.index(h)
        except:
            selected.append(h)
            print("'" + h + "',")

    print("used top features #:", len(selected))

    # filter out features based on prediction model
    X_, headings_ = selectFeatures(X, headings, selected)

    numIterations = 10
    meanMeanError = 0
    for i in range(0, numIterations):
        # use indicies to split matricies randomly but in order
        split = 0.6
        X1, X2, indicies = data.splitMatrixRandomly(X_, split)
        Y1, Y2, indicies = data.splitMatrixRandomly(Y, split, indicies)
        w1, w2, indicies = data.splitMatrixRandomly(weights, split, indicies)

        # flatten data
        Y1 = Y1.flatten()
        Y2 = Y2.flatten()
        w1 = w1.flatten()
        w2 = w2.flatten()

        rf = RandomForestRegressor()
        rf.fit(X1, Y1, w1)
        pred = rf.predict(X2)

        diff = 0
        for p, val in enumerate(pred):
            diff += abs(val - Y2[p])

        meanMeanError += diff / Y2.shape[0]

    print(meanMeanError / numIterations)

    plt.xlim(0, max(Y2))
    plt.ylim(0, max(Y2))
    plt.plot(Y2, Y2, 'bo')
    plt.plot(Y2, pred, 'ro')
    plt.ylabel(str(dv))
    #plt.show()


# -- not used anymore, because data was cleaned so zscore weights lost their purpose
def zscoreplot():
    worker_ratings = data.loadCsvRows(data.FILE_HIT_RATINGS[0], data.FILE_HIT_RATINGS[1])
    hits = data.loadCsvRows("../data/HITs_Ratings_AVG_STD_cleanfiltered.csv", data.FILE_HIT_RATINGS_AVG[1])

    # sort avg overall effort into indexable dictionary
    overall_effort = {}
    column_overall_effort = data.getColumnIndex(hits, "OverallEffort_mean_byExperts")
    column_overall_effort_std = data.getColumnIndex(hits, "OverallEffort_sd_byExperts")
    for hit in hits[1:]:
        overall_effort[hit[0]] = [hit[column_overall_effort], hit[column_overall_effort_std]]


    # sort jobs for each worker into array
    column_overall_effort = data.getColumnIndex(worker_ratings, "Answer.Q_OverallEffort")
    worker_dict = {}
    for rating in worker_ratings[1:]:
        if rating[1] not in worker_dict.keys():
            worker_dict[rating[1]] = []
        worker_dict[rating[1]].append([rating[0], rating[column_overall_effort]])

    # calculate worker specific mean zscore
    worker_zscores = {}
    minNumRatings = 5
    for key in worker_dict.keys():
        if len(worker_dict[key]) < minNumRatings:
            worker_zscores[key] = 1.5
        else:
            num_z_scores = 0
            worker_offset = numpy.zeros((len(worker_dict[key])), dtype=float)
            z_scores = numpy.zeros((len(worker_dict[key])), dtype=float)
            for i, rating in enumerate(worker_dict[key]):

                hasScore = False
                if rating[0] in overall_effort.keys():
                    worker_offset[i] = abs(float(overall_effort[rating[0]][0]) - float(rating[1]))

                    if float(overall_effort[rating[0]][1]) != 0:
                        z_scores[i] = (float(overall_effort[rating[0]][0]) - float(rating[1])) / float(overall_effort[rating[0]][1])
                        z_scores[i] = abs(z_scores[i])
                        num_z_scores += 1
                        hasScore = True

                if not hasScore:
                    z_scores[i] = 0

            worker_zscores[key] = sum(z_scores) / num_z_scores


    hit_dict = {}
    for rating in worker_ratings[1:]:
        if rating[0] not in hit_dict.keys():
            hit_dict[rating[0]] = []
        hit_dict[rating[0]].append(rating[1])


    maxz_score = 0
    hit_zscores = {}
    for key in hit_dict.keys():
        mean_scores = 0
        for worker in hit_dict[key]:
            mean_scores += math.pow(worker_zscores[worker], 2)

        hit_zscores[key] = max(0, 2.5 - mean_scores / len(hit_dict[key]))


    column_overall_effort_std = data.getColumnIndex(hits, "OverallEffort_sd_byExperts") +1
    for h, hit in enumerate(hits):
        if h == 0:
            hit.insert(column_overall_effort_std, "z_score_weighting")
        else:
            hits[h].insert(column_overall_effort_std, hit_zscores[hit[0]])

    data.writeCsv('../data/HITs_Ratings_AVG_STD_zweights.csv', hits)
