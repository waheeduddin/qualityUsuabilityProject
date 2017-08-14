import os
from enum import Enum

import math
import numpy

import time
# -------------------
# --- FEATURES -----
# -------------------
FILE_TAGS = ['../data/tags_formated.csv', ',']
FILE_SEMANTICS = ['../data/semantics.csv', ',']
FILE_IMG = ['valuesForSize50by50', ';']

FILE_HIT_RATINGS = ['../data/HITs_Ratings_final.csv', ',']
FILE_HIT_RATINGS_AVG = ['../data/HITs_Ratings_final_aggregated.csv', ',']

# -------------------
# --- OUTPUT -----
# -------------------
FILE_OUTPUT_AVG = '../data/aggregated/HITS_features_ratings_aggregated.csv'
FILE_OUTPUT_INDIVIDUAL = '../data/aggregated/HITs_features_ratings.csv'




class DV_Type(Enum):
    EFFORT = ['OverallEffort_mean', 'OverallEffort_sd']
    EFFORT_TRIMMED = ['OverallEffort_trimmedMean30', 'OverallEffort_sd']
    COMPLEXITY = ['Answer.Q_COMPLEX_AVG', 'COMPLEX_sd']
    DIFFICULTY = ['DIFFICULT_mean', 'DIFFICULT_sd']

    HIT_TYPE = ['Type', None]
    WOULD_TAKE = ['Answer.Q_TAKE', None]

# ---------------- csv helper functions --------

def loadCsvRows(filename, delim):

    file = open(filename, 'r')

    # if there are two succeeding deliminators the split wont recognize them
    delimX2 = delim + delim
    hasDelimX2 = False

    rows = []
    line = file.readline()
    while line != "":
        # add whitespace in between double delimiters
        while line.find(delimX2) != -1:
            line = line.replace(delimX2, delim + " " + delim)
            hasDelimX2 = True
        # split line into values
        rows.append(line.replace("\n", "").split(delim))
        line = file.readline()

    if hasDelimX2:
        True
        #print("Caution [1]: " + filename + " has empty cells! Make sure rows are parsed correctly")

    # check if all rows have same length
    rowLength = len(rows[0])
    for row in rows:
        if len(row) != rowLength:
            #print('Caution [2]: ' + filename + ' rows have different lengths')
            break

    return rows


def writeCsv(filename, rows):
    file = open(filename, 'w')
    for row in rows:
        strRow = ""
        for i, val in enumerate(row):
            if i > 0:
                strRow += ","
            strRow += str(val)
        strRow += "\n"
        file.write(strRow)
    file.close()


def writeTableHTML(filename, rows, tablename="Table"):
    file = open(filename, 'w')

    file.write('<html><head><title>' + tablename + '</title></head><table border="1">\n')

    for row in rows:
        file.write('<tr>\n')
        for val in row:
            file.write('<td>' + str(val) + '</td>')
        file.write('</tr>\n')

    file.write('</table></body></html>')
    file.close()



# --------- row array helper functions --------------

def getColumnIndex(source, columnName):
    for i, val in enumerate(source[0]):
        if val.find(columnName) != -1:
            return i
    return -1

def findInArrayColumn(rowArray, columnIndex, key):
    for i, row in enumerate(rowArray):
        if key.find(row[columnIndex]) != -1:
            return i
    return -1

def findEqualInArrayColumn(rowArray, columnIndex, key):
    for i, row in enumerate(rowArray):
        if key == row[columnIndex]:
            return i
    return -1

def deleteArrayColumn(source, columnName):
    rowLength = len(source[0])
    columnIndex = source[0].index(columnName)
    for i, row in enumerate(source):
        if columnIndex >= rowLength -1:
            source[i] = row[0:columnIndex]
        else:
            source[i] = row[:columnIndex] + row[columnIndex+1:]

def copyArrayColumn(source, dest, columnIndex):
    for i, row in enumerate(source):
        value = row[columnIndex]
        if len(dest) <= i:
            dest.append([value])
        else:
            dest[i].append(value)


def splitMatrixRandomly(X, split, indicies=None):
    # create data split indicies
    num_train = math.floor(X.shape[0] * split)
    num_test = X.shape[0] - num_train

    # create indicies if none were provided
    if indicies is None:
        indicies = numpy.arange(X.shape[0])
        numpy.random.shuffle(indicies)

    # split data into two matricies
    X1 = numpy.zeros((num_train, X.shape[1]), dtype=float)
    X2 = numpy.zeros((num_test, X.shape[1]), dtype=float)
    for i, index in enumerate(indicies):
        if i < num_train:
            X1[i] = X[index]
        else:
            X2[i - num_train] = X[index]

    return X1, X2, indicies


# ------------- parsing ---------

def noParse(val):
    return val

def parseFloat(val):
    val = val.replace(',', '.')
    if val == "":
        return 0
    elif val.lower() == "y":
        return 1
    elif val.lower() == "n":
        return 0

    try:
        return float(val)
    except:
        return val

def parseHitType(val):
    parseTable = [
        ['SV', 1],
        ['CC', 2],
        ['IA', 3],
        ['IF', 4],
        ['VV', 5],
        ['CA', 6],
        ['UM', 7]
    ]

    for row in parseTable:
        if row[0] == val:
            return row[1]

    return -1

def parseTime(val):
    while val[0] == " ":
        val = val[1:]

    parseTable = [
        ['second', 1],
        ['minute', 60],
        ['hour', 60 * 60],
        ['day', 60 * 60 * 24],
        ['w', 60 * 60 * 24 * 7]
    ]

    try:
        if len(val.split(' ')) < 2:
            val = val.lower().replace('w', ' w').replace('d', ' d').replace('h', ' h').replace('m', ' m')

        split = val.lower().split(' ')
        time = 0
        for i in range(0, len(split)):
            if i % 2 == 0:
                # lookup number multiplication factor
                for entry in parseTable:
                    if split[i + 1].find(entry[0]) != -1:
                        time += entry[1] * float(split[i])
                        break
        return time
    except:
        print('ERROR: cant parse time: ', val)
        return "time"


def parseArrayColumn(source, dest, columnName, parseFunc):
    columnIndex = source[0].index(columnName)
    for i, row in enumerate(source):
        value = row[columnIndex]
        if i > 0:
            value = parseFunc(value)

        # start parsing
        if dest == None:
            source[i][columnIndex] = value
        elif len(dest) <= i:
            dest.append([value])
        else:
            dest[i].append(value)


def parseMultipleArrayColumns(source, parseArray):
    result = []
    for column in parseArray:
        parseArrayColumn(source, result, column[0], column[1])
    return result


def parseArray(source):
    for i, row in enumerate(source):
        for j, val in enumerate(row):
            if i > 0 and j > 0:
                source[i][j] = parseFloat(val)

def parseNestedArray(val, delim, parseFunc, removeAtEnd=0):
    split = val.split(delim)
    split = split[:len(split)-removeAtEnd]
    array = []
    for part in split:
        array.append(parseFunc(part))

    return array


# ------------- Join row arrays ------------

def joinMatricies(arr1, columnIndex1, arr2, columnIndex2, joinFunc):
    result = []
    for i, row in enumerate(arr1):
        # join heading row
        if i == 0:
            result.append(joinFunc(row, arr2[0]))
        # join by arr1 id
        if i > 0:
            index = findEqualInArrayColumn(arr2, columnIndex2, row[columnIndex1].replace(" ", "", -1))
            result.append(joinFunc(row, arr2[index]))

    return result

def joinSoftMatricies(arr1, columnIndex1, arr2, columnIndex2, joinFunc):
    result = []
    for i, row in enumerate(arr1):
        # join heading row
        if i == 0:
            result.append(joinFunc(row, arr2[0]))
        # join by arr1 id
        if i > 0:
            index = findInArrayColumn(arr2, columnIndex2, row[columnIndex1])
            result.append(joinFunc(row, arr2[index]))

    return result

# ---------------- MATRIX functions ----------

def rowsToMatrix(rows, deleteHeadings = False, deleteIndexColumn = -1):

    # delete index colum
    if deleteIndexColumn != -1:
        deleteArrayColumn(rows, rows[0][deleteIndexColumn])

    # delete heading row
    if deleteHeadings:
        rows = rows[1:]

    # turn into numpy matrix
    matrix = numpy.zeros((len(rows), len(rows[0])), dtype=float)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            matrix[i, j] = val

    return matrix


def splitMatrixRandomly(X, split, indicies = None):
    # create data split indicies
    num_train = math.floor(X.shape[0] * split)
    num_test = X.shape[0] - num_train

    # create indicies if none were provided
    if indicies is None:
        indicies = numpy.arange(X.shape[0])
        numpy.random.shuffle(indicies)

    # split data into two matricies
    X1 = numpy.zeros((num_train, X.shape[1]), dtype=float)
    X2 = numpy.zeros((num_test, X.shape[1]), dtype=float)
    for i, index in enumerate(indicies):
        if i < num_train:
            X1[i] = X[index]
        else:
            X2[i - num_train] = X[index]


    return X1, X2, indicies


# ------------ LOADING functions ------------

def loadImgFeatures():
    # load image feature csv file
    imgRows = loadCsvRows('../data/img/' + FILE_IMG[0] + '.txt', FILE_IMG[1])

    densities = [[]]
    histogram = [[]]

    # FORMAT and REARRANGE DATA: to fit into data structure
    for i, row in enumerate(imgRows):
        if i > 0:
            # remove .png extension from imageID
            row[0] = row[0].split('.')[0]

            # [3] convert NaN to zero
            if row[3] == "NaN":
                row[3] = "0"

            densities.append(parseNestedArray(imgRows[i][1], ',', noParse, 1))
            histogram.append(parseNestedArray(imgRows[i][2], ',', noParse, 1))

        # rearrange
        imgRows[i] = row[0:1] + row[3:8]


    # add histogram and densities into imgRows
    appendix = [
        [histogram, "histogram_"],
        #[densities, "densities_"]
    ]
    for row in appendix:
        arr = row[0]
        name = row[1]
        for i in range(0, len(arr[1])):
            arr[0].append(name + str(i))

        for i in range(0, len(arr[1])):
            copyArrayColumn(arr, imgRows, i)

    deleteArrayColumn(imgRows, 'largest connected window in pixels')

    additionalRows = loadCsvRows('../data/img_additional/' + FILE_IMG[0] + '.csv', ',')
    imgRows = joinMatricies(imgRows, 0, additionalRows, 0, lambda row1, row2: row1 + row2[1:])

    return imgRows

def mergeRows(row1, row2):
    return row1

def loadExtractedFeatures():
    # load image features
    images = loadImgFeatures()
    parseArray(images)
    # load html features
    html = loadCsvRows(FILE_TAGS[0], FILE_TAGS[1])
    parseArray(html)
    # load semantic features
    semantics = loadCsvRows(FILE_SEMANTICS[0], FILE_SEMANTICS[1])
    parseArray(semantics)

    features = joinSoftMatricies(images, 0, html, 0, lambda row1, row2: row1 + row2[1:])
    features = joinSoftMatricies(features, 0, semantics, 0, lambda row1, row2: row1 + row2[1:])

    return features

def loadAvgFeaturesAndRatings(nameLabel, nameLabelSd):
    # load HIT data
    hitRows = loadCsvRows(FILE_HIT_RATINGS_AVG[0], FILE_HIT_RATINGS_AVG[1])

    # replace whitespace in imageID - why is it even in there
    for r, rating in enumerate(hitRows):
        hitRows[r][0] = rating[0].replace(" ", "", -1)

    # parse used HIT array columns
    hits = parseMultipleArrayColumns(hitRows, [
        ["imageID", noParse],
        ["Input.timeAlloted", parseTime],
        ["Input.reward", parseFloat],
        ["Input.hitsAvailable", parseFloat],
        ['Type', parseHitType]
    ])

    # join extracted features for all 400 HITs
    extracted = loadExtractedFeatures()
    features = joinMatricies(hits, 0, extracted, 0, lambda row1, row2: row1 + row2[1:])

    # filter features and ratings, only select HITs used in hitRows
    ratings = joinMatricies(hitRows, 0, features, 0, lambda row1, row2: row1)
    features = joinMatricies(ratings, 0, features, 0, lambda row1, row2: row2)

    # copy dependent variable into labels
    labels = []
    copyArrayColumn(ratings, labels, getColumnIndex(ratings, nameLabel))
    # copy standard deviation for dependent variable into sds
    sds = []
    if nameLabelSd != None:
        copyArrayColumn(ratings[1:], sds, getColumnIndex(ratings, nameLabelSd))

    # safe features and ratings as one table
    output = joinMatricies(features, 0, ratings, 0, lambda row1, row2: row2 + row1)
    writeCsv(FILE_OUTPUT_AVG, output)

    return features, labels, sds


def loadIndividualFeaturesAndRatings(nameLabel, nameLabelSd):
    # load hit ratings and worker survey values
    hits = loadCsvRows(FILE_HIT_RATINGS[0], FILE_HIT_RATINGS[1])
    deleteArrayColumn(hits, 'WorkerId')

    # join survey and ratings together and parse
    features = hits
    parseArray(features)

    # load extracted features and join
    extracted = loadExtractedFeatures()
    features = joinMatricies(features, 0, extracted, 0, lambda row1, row2: row1 + row2[1:])

    # safe features and ratings together to have an output
    writeCsv(FILE_OUTPUT_INDIVIDUAL, features)
    time.sleep(5)
    # copy labels out and remove from features
    labels = []
    takeIndex = getColumnIndex(features, nameLabel)
    copyArrayColumn(features, labels, takeIndex)
    deleteArrayColumn(features, nameLabel)

    # copy standard deviation for labels out and remove from features
    sds = []
    if nameLabelSd != None:
        takeIndex = getColumnIndex(features, nameLabelSd)
        copyArrayColumn(features, sds, takeIndex)
        deleteArrayColumn(features, nameLabelSd)

    return features, labels, sds


def convertDataIntoMatrix(features, ratings, variances):
    # clean features
    deleteArrayColumn(features, 'imageID')  # remove indexColumn
    headings = features[0]  # save headings into array
    features = features[1:]  # delete heading row

    # copy features into numpy array
    data = numpy.zeros((len(features), len(features[0])), dtype=float)
    for r, row in enumerate(features):
        for i, val in enumerate(row):
            data[r, i] = val

    # copy labels into numpy array
    labels = numpy.zeros((len(ratings) - 1, 1), dtype=float)
    for i in range(0, labels.shape[0]):
        labels[i, 0] = ratings[1 + i][0]

    # copy variances into numpy array
    weights = numpy.zeros((len(variances), 1), dtype=float)
    for i in range(0, weights.shape[0]):
        weights[i, 0] = float(variances[i][0])

    return data, labels, weights, headings




def loadAggregatedFeaturesAndLabels(dv_type):
    features, ratings, ratingVariance = loadAvgFeaturesAndRatings(dv_type.value[0], dv_type.value[1])
    data, labels, weights, headings = convertDataIntoMatrix(features, ratings, ratingVariance)

    return data, labels, weights, headings

def loadIndividualFeaturesAndLabels(dv_type):
    features, ratings, ratingVariance = loadIndividualFeaturesAndRatings(dv_type.value[0], dv_type.value[1])
    data, labels, weights, headings = convertDataIntoMatrix(features, ratings, ratingVariance)

    return data, labels, weights, headings


def loadFeaturesAndLabels(dv_type):
    if dv_type != DV_Type.WOULD_TAKE:
        return loadAggregatedFeaturesAndLabels(dv_type)
    else:
        return loadIndividualFeaturesAndLabels(dv_type)
