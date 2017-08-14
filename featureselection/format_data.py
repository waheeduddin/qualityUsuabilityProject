import os
import data


# --- disclaimer: hacked together script to replace some badly formated .csv files
# probably shouldn't be used anywhere else

def formatTagsDsv():
    oldCsv = open('../data/tags.csv')

    # read first line
    line = oldCsv.readline()
    newDoc = line

    while line != "":
        line = oldCsv.readline()

        if line.find('[') != -1:
            # parse until bracket
            pre = line[:line.find('[')]
            line = line[line.find('['):]

            indexBracketClose = line.find(']')
            indexComma = line.find(',')
            while indexComma < indexBracketClose:
                line = line.replace(',', ';', 1)
                indexComma = line.find(',')

            newDoc += pre + line
        else:
            newDoc += line


    file = open('../data/tags_formated.csv', 'w')
    file.write(newDoc)
    file.close()


def formatAdditionalImgFeatures():

    imgdir = "../data/img_additional/"
    for name in os.listdir(imgdir):
        oldCsv = open(imgdir + name)

        # read first line
        line = oldCsv.readline().replace('"Image Ids"', "imageID")
        newDoc = line


        while(line != ""):
            line = oldCsv.readline().replace("\n", "").replace(".png", "").replace('"', "", -1)
            line2 = oldCsv.readline().replace('"', "", -1).replace("-1.0", "0")

            newDoc += line + line2



        file = open(imgdir + name, 'w')
        file.write(newDoc)
        file.close()


def joinHits():
    hits = data.loadCsvRows('../data/HITs_Ratings_AVG_STD.csv', ',')
    for i, row in enumerate(hits):
        hits[i][0] = str(row[0]).replace(' ', '', -1)

    ratings = data.loadCsvRows('../data/HITs_Ratings_noTLX.csv', ',')
    output = data.joinMatricies(hits, 0, ratings, 0, lambda row1, row2: row1[0:1] + row2[2:5] + row1[1:])
    data.writeCsv('../data/HITs_Ratings_AVG_STD.csv', output)


def removeAritifialHits():
    oldCsv = open('../data/HITs_Ratings_AVG_STD_filtered.csv')

    # read first line
    line = oldCsv.readline()
    newDoc = line

    while line != "":
        line = oldCsv.readline()

        if line.split(',')[0].find('_') == -1:
            newDoc += line

    file = open('../data/HITs_Ratings_AVG_STD_cleanfiltered.csv', 'w')
    file.write(newDoc)
    file.close()


#joinHits()
#formatTagsDsv()
#formatAdditionalImgFeatures()
removeAritifialHits()