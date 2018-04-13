import random
import numpy as np


def readOriginalFile():
    '''
    get a file like fasta form
    :return:
    '''
    f = open("Acetylation.elm", "r")
    # f2=open("test/proteinId.txt", "w", encoding="utf8")
    # f3=open("test/protein.txt", "w", encoding="utf8")
    f4 = open("allProtein.txt", "w", encoding="utf8")
    f.readline()
    i = 1
    name = ""
    for line in f:
        splitedResult = line.split(sep="\t")
        proteinName = splitedResult[1]
        if proteinName == name:
            continue
        else:
            name = proteinName
            # f2.write(">"+splitedResult[1]+"\n")
            # f3.write(splitedResult[4]+"\n+"\n")
            f4.write(">"+splitedResult[1]+"\n")
            f4.write(splitedResult[4]+"\n")
        i = i+1
    print("finished")


def getProteinSite():
    f = open("Acetylation.elm", "r")
    f.readline()
    name = ""
    for line in f:
        splitedResult = line.split(sep="\t")
        proteinName = splitedResult[1]
        if proteinName == name:
            fw.write(splitedResult[2]+" ")
            continue
        else:
            fw = open("e:/12/" + proteinName, "w", encoding="utf8")
            name = proteinName
            fw.write(splitedResult[2]+" ")
    print("finished")


def getProteinWithSite():
    '''
    get a file like fasta form and have 01 flag
    i call it as "allProteinN01.txt"
    :return:
    '''
    f = open("Acetylation.elm", "r")
    f4 = open("allProteinN01.txt", "w", encoding="utf8")
    f.readline()
    name = ""
    numberSequence = [0]
    for line in f:
        splitedResult = line.split(sep="\t")
        proteinName = splitedResult[1]
        lengthOfSequence = len(splitedResult[4])
        if proteinName == name:
            index = int(splitedResult[2])
            numberSequence[index-1] = 1
            continue
        else:
            name = proteinName
            if numberSequence == [0]:
                f4.write(">" + splitedResult[1] + "\n")
                f4.write(splitedResult[4] + "\n")
                numberSequence = [0]*lengthOfSequence
                index = int(splitedResult[2])
                numberSequence[index - 1] = 1
            else:
                ns = str(numberSequence).strip('[').strip(']').replace(', ', '')
                f4.write(ns+'\n')
                f4.write(">" + splitedResult[1] + "\n")
                f4.write(splitedResult[4] + "\n")
                numberSequence = [0]*lengthOfSequence
                index = int(splitedResult[2])
                numberSequence[index - 1] = 1
    ns = str(numberSequence).strip('[').strip(']').replace(', ', '')
    f4.write(ns + '\n')

def separateFile():
    '''
    separate a consistent file into a series of single fasta-form files
    :return:
    '''
    f = open("allProteinN01.txt", "r")
    while 1:
        line = f.readline()
        if not line:
            break
        f2 = open("e:/11/"+line.strip().strip('>'), "w", encoding="utf8")
        f2.write(line)
        line = f.readline()
        f2.write(line)
        line = f.readline()
        f2.write(line)
        f2.close()

def makeFastafileInto01Fasta(fastaFileName, objectFileName):
    '''
    get de-redundancy fasta file into a form with 01 flag, like below:
    >asdfd
    SASDFKRKWER
    00000001000
    :param fastaFileName:
    :param objectFileName:
    :return:
    '''
    f = open(fastaFileName, "r")
    fo = open(objectFileName, "w", encoding="utf8")
    while 1:
        line = f.readline()
        if not line:
            break
        name = line.strip().strip('>')
        f2 = open("e:/11/"+name, "r")
        for i in range(3):
            line2 = f2.readline()
            fo.write(line2)
        f2.close()
        f.readline()
    f.close()
    fo.close()

def shuffleTwoArray(allData, *random_seed):
    if random_seed != ():
        random.seed(random_seed)
    data1 = allData[0]
    data2 = allData[1]
    if len(data1) != len(data2):
        raise
    index = list(range(0, len(data1)))
    rs = random.sample(index, len(index))
    data1 = np.array(data1)[rs]
    data2 = np.array(data2)[rs]
    return data1, data2



def shuffleArray(allData, *random_seed):
    if random_seed != ():
        random.seed(random_seed)
    index = list(range(0, len(allData)))
    rs = random.sample(index, len(index))
    allData = np.array(allData)[rs]
    return allData


def makeSequenceFileIntoFasta(fileName):
    fr = open(fileName, "r")
    fw = open(fileName.replace("seq", "fasta"), 'w', encoding="utf8")
    order = 0
    while 1:
        line = fr.readline()
        if not line:
            print(fileName, " transfer over")
            break
        fw.write('>'+fileName[7:10]+str(order)+'\n')
        fw.write(line)
        order = order+1
    fr.close()
    fw.close()


def makeFastaFileIntoSequence(fileName):
    fr = open(fileName, "r")
    fw = open(fileName.replace("fasta", "seq"), 'w', encoding="utf8")
    while 1:
        line = fr.readline()
        if not line:
            break
        line = fr.readline()
        fw.write(line)
    fr.close()
    fw.close()

def getProteinAndName(originalFileName, objectFileName):
    '''
    get file and its form like below:
        >asdfd
        SASDFKRKWER
    '''
    f = open(originalFileName, "r")
    f4 = open("objectFileName", "w", encoding="utf8")
    f.readline()
    i = 1
    name = ""
    for line in f:
        splitedResult = line.split(sep="\t")
        proteinName = splitedResult[1]
        if proteinName == name:
            continue
        else:
            name = proteinName
            f4.write(">" + splitedResult[1] + "\n")
            f4.write(splitedResult[4] + "\n")
        i = i + 1
    print("finished")

def getProteinWithSiteAndName(originalFileName, objectFileName):
    '''
    get file and its form like below:
        >asdfd
        SASDFKRKWER
        00000001000
    '''
    f = open(originalFileName, "r")
    f4 = open(objectFileName, "w", encoding="utf8")
    f.readline()
    name = ""
    numberSequence = [0]
    for line in f:
        splitedResult = line.split(sep="\t")
        proteinName = splitedResult[1]
        lengthOfSequence = len(splitedResult[4])
        if proteinName == name:
            index = int(splitedResult[2])
            numberSequence[index - 1] = 1
            continue
        else:
            name = proteinName
            if numberSequence == [0]:
                f4.write(">" + splitedResult[1] + "\n")
                f4.write(splitedResult[4] + "\n")
                numberSequence = [0] * lengthOfSequence
                index = int(splitedResult[2])
                numberSequence[index - 1] = 1
            else:
                ns = str(numberSequence).strip('[').strip(']').replace(', ', '')
                f4.write(ns + '\n')
                f4.write(">" + splitedResult[1] + "\n")
                f4.write(splitedResult[4] + "\n")
                numberSequence = [0] * lengthOfSequence
                index = int(splitedResult[2])
                numberSequence[index - 1] = 1

    ns = str(numberSequence).strip('[').strip(']').replace(', ', '')
    f4.write(ns + '\n')

def extractWindowSizeSequence(sequence, index, windowSize):
    '''
    deal a single sequence, always be the sub-program of getPartSequence
    '''
    partSequence = ''
    if (index - windowSize) < 0 and (index + windowSize) > len(sequence):
        for i in range(windowSize - index):
            partSequence += 'B'
        for i in range(0, len(sequence)):
            partSequence += sequence[i]
        for i in range(len(sequence), index + windowSize + 1):
            partSequence += 'B'
    elif (index - windowSize) < 0 and (index + windowSize + 1) <= len(sequence):
        for i in range(windowSize - index):
            partSequence += 'B'
        for i in range(0, index + windowSize + 1):
            partSequence += sequence[i]
    elif (index - windowSize) >= 0 and (index + windowSize + 1) > len(sequence):
        for i in range(index - windowSize, len(sequence)):
            partSequence += sequence[i]
        for i in range(len(sequence), index + windowSize + 1):
            partSequence += 'B'
    elif (index - windowSize) >= 0 and (index + windowSize + 1) <= len(sequence):
        for i in range(index - windowSize, index + windowSize + 1):
            partSequence += sequence[i]
    return partSequence

def getPartSequence(fasta01FileName, windowSize, windowType, Train_or_Test):
    '''
    will get two files PosSequence and NegSequence where each line is a sequence part.
    :param fastaFileName: form is with 01 and like "Trainset.txt"
    :param windowSize:
    :param fileType:Train or Test
    :return:
    '''
    f = open(fasta01FileName, "r")
    fPos = open("seq/"+Train_or_Test+windowType+"PosSequence.seq", "w", encoding="utf8")
    fNeg = open("seq/"+Train_or_Test+windowType+"NegSequence.seq", "w", encoding="utf8")
    i = 0
    while 1:
        line = f.readline()
        if not line:
            break
        sequence = f.readline().strip()
        flags = f.readline().strip()
        for i in range(len(sequence)):
            if sequence[i] == 'K':
                partSequence = extractWindowSizeSequence(sequence=sequence, index=i, windowSize=windowSize)
                if flags[i] == '1':
                    fPos.write(partSequence+"\n")
                elif flags[i] == '0':
                    fNeg.write(partSequence+"\n")
    # print("separate positive sequence and negative sequence is okay")


def getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir=""):
    fasta01FileName = fastaFileName.replace("fasta", "txt")
    makeFastafileInto01Fasta(fastaDir+fastaFileName, fastaDir+fasta01FileName)
    getPartSequence(fastaDir+fasta01FileName, windowSize, windowType, Train_or_Test)


if __name__ == '__main__':
    # fastaDir = "fasta/"
    # fastaFileName = "TrainSet.fasta"
    # Train_or_Test = "Train"
    # windowType = "OOK"
    # windowSize = 7
    # getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    makeSequenceFileIntoFasta("seq/TestAAIndexPosSequence.seq")
    makeSequenceFileIntoFasta("seq/TestAAIndexNegSequence.seq")
    # fasta01FileName = fastaFileName.replace("fasta", "txt")
    # makeFastafileInto01Fasta(fastaDir+fastaFileName, fastaDir+fasta01FileName)
    # getPartSequence(fastaDir+fasta01FileName, windowSize, windowType,Train_or_Test)
    # getPartSequence(fastaDir+"musculus_sspka.txt", windowSize=26, windowType="OOK", Train_or_Test="Test")
    # getPartSequence(fastaDir+"musculus_sspka.txt", windowSize=14, windowType="AAIndex", Train_or_Test="Test")




