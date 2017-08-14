####
# FeatureExtractor v 0.0.1 
# Goal is to extract features from HTML file that contains the HIT.
# Supported features:
#    - Count number of images
#       - Count number of a, p, div, button, video tags
#       - Count number of inputs by type and 'required' attributes
#       - Count number of characters
#       - Count number of bonus occurrences
#       - Count number of headers by size
#       - Measure the page length in pixels
#       - Convert visible text to stems
#       - Save the information above in CSV files
#
# Dependencies: webkit2png
####
import re
import csv
import codecs
from os import listdir, getcwd, remove, chdir
from os.path import isfile, join, abspath,getsize,isabs
import subprocess as sub
from collections import defaultdict
from nltk import PorterStemmer
from bs4 import BeautifulSoup
#import Image
import urllib
import io
import sys,traceback

tagsFile = None
tagsFileWriter = None
stemsFile = None
stemsFileWriter = None
semanticsFile = None
semanticsWriter = None

tagsHeader = None

def is_absolute(url):
    return bool(urllib.parse.urlparse(url).netloc)

def openCSVFile(fileName, header=None):
    csvFile = open(fileName+'.csv', 'w', newline='')
    csvFileWriter = csv.writer(csvFile, delimiter=',')
    if type(header) == list:
        csvFileWriter.writerow(header)
        
    return csvFile, csvFileWriter

def closeCSVFile(file):
    if file:
        file.close()

def getHTMLFilesFromDirectory(directory):
    htmlFiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and f.endswith(".html"))]
    return htmlFiles

def countChars(htmlContent):
    # Characters count
    text = htmlContent.findAll(text=True)
    filteredText = "".join([x for x in text if visible(x)])
    charCount = len(filteredText)
    
    return charCount

def countTags(htmlContent, groupID):
    global tagsHeader
    # List of tags we want to count
    tagsList = ['bonus', 'characters', 'img', 'input', 'header', 'textarea', 'selection', 'p', 'a', 'div', 'span', 'button', 'video', 'length','fileSize','z_sum_inputs']

    # List of image attributes we want to count
    imgAttrs = ['count', 'over100px', 'under100px']

    # List of input attributes we want to count
    inputAttrs = ['count', 'type', 'required']

    # List of textarea attributes
    textareaAttrs = ['count','size']

    # List of 'a' attributes
    aAttrs = ['count', 'innerHref(#)']

    # List of header sizes
    headerSizes = ['h%i' % k for k in range(6) if k>0]

    # List of input types
    inputTypes = ['button', 'checkbox', 'color', 'date', 'datetime', 'datetime-local', 'email', 'file', 'hidden', 'image', 'month', 'number', 'password', 'radio', 'range', 'reset', 'search', 'submit', 'tel', 'text', 'text.readonly', 'time', 'url', 'week']
    #inputTypes = ['hidden', 'radio']

    # Dictionary with count of each tag
    tagCount = {tag:0 for tag in tagsList}
    tagCount['input'] = {x:0 for x in inputAttrs}
    tagCount['a'] = {j:0 for j in aAttrs}
    tagCount['input']['type'] = {y:0 for y in inputTypes}
    tagCount['img'] = {z:0 for z in imgAttrs}
    tagCount['header'] = {b:0 for b in headerSizes}
    tagCount['textarea'] = {n:0 for n in textareaAttrs}

    tagsHeader = [['%s.%s'% (str(y),str(z)) for z in list(tagCount[y].keys())] for y in list(tagCount.keys()) if isinstance(tagCount[y], dict)]
    tagsHeader.append(['input.type.'+str(t) for t in inputTypes])
    tagsHeader.append([s for s in list(tagCount.keys()) if not isinstance(tagCount[s], dict)])
    tagsHeader = [val for sublist in tagsHeader for val in sublist]
    tagsHeader.remove('input.type')
    tagsHeader.sort()
    tagsHeader.insert(0, 'groupID')
    
    for tagName in tagsList:
        if tagName == 'fileSize':
            size=getsize(groupID+'.html')
            tagCount['fileSize']=size
        elif tagName == 'bonus':
            text = htmlContent.findAll(text=True)
            filteredText = list(filter(visible, text))
            bonusCount = sum(('bonus' in i or '$' in i) for i in filteredText)
            tagCount['bonus'] = bonusCount
            
        elif tagName == 'characters':
            tagCount['characters'] = countChars(htmlContent)

        elif tagName == 'header':
            for size in headerSizes:
                count = len(htmlContent.find_all(size))
                tagCount['header'][size] = count

        elif tagName == 'selection':
            tagCount['selection'] = len(htmlContent.find_all('select'))

        elif tagName == 'textarea':
            textareas = htmlContent.find_all('textarea')
            tagCount['textarea']['count'] = len(textareas)
            sizes = []
            for txt in textareas:
                try:
                    sizes.append('%s*%s' % (txt['rows'], txt['cols']))
                except: # No rows or cols attributes
                    pass
            tagCount['textarea']['size'] = sizes
            
        elif tagName == 'input':
            allInputs = htmlContent.find_all('input')
            readonly = 0
            for typ in inputTypes:
                result = []
                names = []
                required = 0
#                print ("*** looking for: "+typ)
                for f in allInputs:
#                    print (f.attrs, end="")
#                    print ('type'in f.attrs, end="")
#                    input("(IN)")
                    if 'type'in f.attrs and f['type'] == typ:
#                        print (" t:"+f['type']+"\t", end="")
#                    if f['type'] == typ:
                        if typ == 'text':
                            try:
                                if f['readonly'] == 'readonly':
                                    readonly += 1
                            except:
                                pass
                        try:
                            # Check for unique input names
                            if f['name'] not in names:
                                result.append(f)
                                names.append(f['name'])
                                try:
                                    if f['required']:
                                        required += 1
                                except KeyError:
                                    # Input has no 'required' attribute
                                    pass
                        except KeyError:
                            # Input has no name
                            result.append(f)
                    #print("OK")
                tagCount['input']['type'][typ] = len(result)
                tagCount['input']['required'] = required
            tagCount['input']['type']['text.readonly'] = readonly
            tagCount['input']['count'] = len(allInputs)
        elif tagName=='z_sum_inputs': 
            s=0;
            for typ in inputTypes:
                s=s+tagCount['input']['type'][typ]
            s=s-tagCount['input']['type']['hidden']
            s=s-tagCount['input']['type']['button']
            s=s-tagCount['input']['type']['reset']
            s=s-tagCount['input']['type']['submit']
            s=s-tagCount['input']['type']['text.readonly']
            tagCount['z_sum_inputs'] = s+tagCount['textarea']['count']+tagCount['selection']
        elif tagName == 'a':
            allATags = htmlContent.find_all('a')
            tagCount['a']['count'] = len(allATags)

            hrefCount = 0
            for a in allATags:
                try:
                    if (a['href'] == "#") or (a['href'].startswith('#')):
                        hrefCount += 1
                except:
                    pass
            tagCount['a']['innerHref(#)'] = hrefCount

        elif tagName == 'img':
            allImgTags = htmlContent.find_all('img')
            tagCount['img']['count'] = len(allImgTags)
            print ('Images ('+ str(tagCount['img']['count'])+"):", end="")
            over100px = 0
            under100px = 0
            w = h = 0
            for q in allImgTags:
                try:
                    try:
                        s = ''.join(x for x in q['height'] if x.isdigit())
                        h = int(s)
                    except:
                        pass
                    s = ''.join(x for x in q['width'] if x.isdigit())
                    w = int(s)
                except:
                    try:
                        print ('.', end="")
                        # avoid cases that image is written in SRC as BASE64
                        if len(q['src'])>1 and len(q['src'])<100: 
                            w=h=-1
                            if is_absolute(q['src']) and q['src'].index("//")!=0:
                                tmpFileName="testXYZ00000001.jpg"
                                fd = urllib.request.urlopen(q['src'])
                                image_file = io.BytesIO(fd.read())
                                if q['src'].index("//")==0:
                                    q['src']=q['src'].replace("//","http://",1)
                                urllib.request.urlretrieve(q['src'],tmpFileName )
                                #im = Image.open(tmpFileName)
                                #w, h = im.size
                                #im.close()
                                remove(tmpFileName)
                            else:
                                #im = Image.open(q['src'])
                                #w, h = im.size
                                #im.close()
                                menodo = True
                            if (w!=-1):
                                if (((w >= 100) or (h >= 100))):
                                    over100px += 1
                                else:
                                    under100px += 1
                    except Exception as e:
                        print ('Exception error is: %s' % e)
                        pass
            tagCount['img']['over100px'] = over100px
            tagCount['img']['under100px'] = under100px
            print ("\t", end="")

        elif tagName == 'length':
           # try:
           #    sub.run('wkhtmltoimage -q --load-error-handling ignore --javascript-delay 50 %s.html %s.png' % (groupID, groupID), shell=True, check=True)
           # except:
            #   pass
            
           # if isfile(groupID+'.png'):
            #    im = Image.open('%s.png' % groupID)
            #    width, height = im.size
            #    tagCount['length'] = '%i*%i' % (width, height)
            #else:
            #    tagCount['length'] = '0*0'
            # --- from Rafael
            #sub.run('webkit2png -o %s -F ~/Documents/TelekomLab/Turkmotion/Study3/HTML/Test/%s.html' % (groupID, groupID), shell=True, check=True, stdout=sub.PIPE)
            #im = Image.open('%s-full.png' % groupID)
            #width, height = im.size
            tagCount['length'] = 'nn'
        else:
            try:
                tagCount[tagName]['count'] = len(htmlContent.find_all(tagName))
            except:
                tagCount[tagName] = len(htmlContent.find_all(tagName))

    return tagCount

def getUniqueStems(stemDictArray, id):

    uniques = []
    words = stemDictArray[id].keys()

    for word in words:
        unique = True
        for i, dict in enumerate(stemDictArray):
            if i != id and word in dict.keys():
                unique = False
                break
        if unique:
            uniques.append(word)

    return uniques

def semanticFeatures(htmlContent):
    # For removing non alphabetic characters from words
    regex = re.compile('[^a-zA-Z]')
    text = htmlContent.findAll(text=True)
    # Separate text into words
    filteredText = list(filter(visible, text))

    # init semantics dict
    semantics = {}
    semantics['numSentTotal'] = 0
    semantics['numSubTotal'] = 0
    semantics['numWordsTotal'] = 0
    semantics['avgWordLength'] = 0

    # find word stems and calc average
    words = re.findall(r"[\w']+", regex.sub(' ', str(filteredText)))
    for w in words:
        semantics['avgWordLength'] += len(w)
    semantics['avgWordLength'] /= len(words)

    for text in filteredText:
        numWords = text.count(" ")

        # only counts as a sentence if there are more than 3 whitespaces
        if numWords > 3:
            # add number of words to total
            semantics['numWordsTotal'] += numWords

            # estimate num sentences and add to total
            numSent = text.count(".") + text.count(";")
            if numSent > 0:
                semantics['numSentTotal'] += numSent
            else:
                semantics['numSentTotal'] += 1

            # add subclauses to total
            semantics['numSubTotal'] += text.count(",") + text.count("(")


    semantics['avgNumWords'] = semantics['numWordsTotal'] / semantics['numSentTotal']


    return semantics


def countStems(htmlContent, stemDictArray):
    # For removing non alphabetic characters from words
    regex = re.compile('[^a-zA-Z]')
    
    text = htmlContent.findAll(text=True)

    # Separate text into words
    filteredText = list(filter(visible, text))
    words = re.findall(r"[\w']+", regex.sub(' ', str(filteredText)))

    # List of words to ignore - 100 most common words in English
    ignore = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us']

    #words = [regex.sub(' ', i) for i in words]
    words = [w for w in words if w not in ignore]

    # Create dictionary with integer values for counting stems
    stemCount = defaultdict(int)

    return 0
    dict = {}
    for word in words:
        # Convert word to stem
        stem = PorterStemmer().stem_word(word)
        dict[stem] = 1

        # Store stem count
        stemCount[stem] += 1

    stemDictArray.append(dict)
    return stemCount

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True

def storeStems(fileId, stemDict):

    global stemsFile
    global stemsFileWriter
    
    if not stemsFile:
        stemsFileHeader = ['groupID', 'stem', 'count']
        stemsFile, stemsFileWriter = openCSVFile('stems', stemsFileHeader)

    for stem in list(stemDict.keys()):
        row = [fileId, stem, stemDict[stem]]
        stemsFileWriter.writerow(row)
        
    stemsFile.flush()

def dictValuesToList(dictIn, listOut):
    for key, value in dictIn.items():
        if isinstance(value, dict): # If value itself is a dictionary
            dictKeysToList(value, listOut)
        else:
            listOut.append(str(value))
    return listOut

def storeTags(fileId, tagDict):

    global tagsFile
    global tagsFileWriter
    global tagsHeader

    if not tagsFile:
        tagsFile, tagsFileWriter = openCSVFile('tags', tagsHeader)

    row = [fileId]

    for i in tagsHeader:
        if i == 'groupID':
            continue
        elementToAppend = 0
        if '.' in i:
            elements = i.split('.')
            if len(elements) < 3:
                elementToAppend = str(tagDict[elements[0]][elements[1]])
            elif len(elements) == 4:
                joinLast = '.'.join((elements[2], elements[3]))
                elementToAppend = str(tagDict[elements[0]][elements[1]][joinLast])
            else:
                elementToAppend = str(tagDict[elements[0]][elements[1]][elements[2]])
        else:
            elementToAppend = str(tagDict[i])
        row.append(elementToAppend)
        
    tagsFileWriter.writerow(row)
    tagsFile.flush()



def storeSemantics(fileId, semantics):


    global semanticsFile
    global semanticsFileWriter

    if not semanticsFile:
        semanticsFileHeader = ['GroupId', 'Sentences', 'Subclauses', 'Words', 'AvgWordLength', 'numUniqueStems', 'avgUniqueStems']
        semanticsFile, semanticsFileWriter = openCSVFile('semantics', semanticsFile)
        semanticsFileWriter.writerow(semanticsFileHeader)

    semanticsFileWriter.writerow([fileId,
        semantics['numSentTotal'],
        semantics['numSubTotal'],
        semantics['numWordsTotal'],
        semantics['avgWordLength'],
        semantics['numUniqueStems'],
        semantics['avgUniqueStems'],
    ])
    semanticsFile.flush()

htmlDir = '../data/sources/'
saveDir = getcwd()
files = getHTMLFilesFromDirectory(htmlDir)

stemDicts = []
semanticsArray = []

for counter, f in enumerate(files):
    try:
        if (counter < 1):
            # ------> load html
            chdir(htmlDir)
            print (str(counter) + "\t", end="")
            htmlFile = abspath(f)
            htmlFileID = f.split('.')[0]
            print (htmlFileID+ "\t", end="")      
            htmlContent = BeautifulSoup(codecs.open(htmlFile, "r", "utf-8"), "html.parser")
            print ( "Loaded \t", end="")
            tags = countTags(htmlContent, htmlFileID)
            chars = countChars(htmlContent)
            stems = countStems(htmlContent, stemDicts)
            semanticsArray.append(semanticFeatures(htmlContent))

            # store features
            chdir(saveDir)
            storeTags(htmlFileID, tags)
            storeStems(htmlFileID, stems)

            print("OK")
    except Exception as e:
        print ('**** Exception error is: %s' % e)
        #traceback.print_exc()
        pass


print("-> extracting semantics")
for counter, f in enumerate(files):
    try:
        if (counter < 0):
            uniqueStems = getUniqueStems(stemDicts, counter)
            semantics = semanticsArray[counter]
            semantics['numUniqueStems'] = len(uniqueStems)
            semantics['avgUniqueStems'] = semantics['numUniqueStems'] / semantics['numWordsTotal']
            storeSemantics(f.split('.')[0], semantics)

    except Exception as e:
        print ('**** Exception error is: %s' % e)
        #traceback.print_exc()
        pass
