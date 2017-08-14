import data
import numpy

import matplotlib.pyplot as plt


def calcCorrelationEffortTLX():
    ratings = data.loadCsvRows('../data/HITs_Ratings_TLX.csv', ',')

    index_effort = -1
    index_tlx = -1
    index_tlxraw = -1
    for i, val in enumerate(ratings[0]):
        if val == "Answer.Q_OverallEffort":
            index_effort = i
        elif val == "TLX":
            index_tlx = i
        elif val == "TLX_RAW":
            index_tlxraw = i


    effort = numpy.zeros((len(ratings[1:])), dtype=float)
    tlx = numpy.zeros((len(ratings[1:])), dtype=float)
    tlx_raw = numpy.zeros((len(ratings[1:])), dtype=float)

    for i, row in enumerate(ratings[1:]):
        effort[i] = ((float(row[index_effort]) / 150) * 100)
        tlx[i] = (row[index_tlx])
        tlx_raw[i] = (row[index_tlxraw])


    print("correlation tlx-eff: ", numpy.corrcoef(tlx, effort)[0, 1])
    print("correlation tlxraw-eff: ", numpy.corrcoef(tlx_raw, effort)[0, 1])

    plt.xlabel("TLX Scores")
    plt.ylabel("Overall Effort Rating")
    plt.plot(tlx, effort, 'ro')
    plt.show()


calcCorrelationEffortTLX()