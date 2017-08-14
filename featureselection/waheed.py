


with open("imageFiles.txt", "r") as ins:
    array = []
    for line in ins:
        array.append(line)
for i in range(0, (len(array))):
    replaces = array[i].replace("\n", "");
#    print(replaces + ".txt")

FILE_IMG2 = ['valuesForSize50by50', ';']
FILE_IMG = [replaces, ';']

print(type(FILE_IMG[0]))
print(type(FILE_IMG2[0]))
