import random
import numpy as np
import pandas as pd
from InitAAindex import InitAAindex
from tools import *

class InitFeature(object):

    def getProteinAndName(self, originalFileName, objectFileName):
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


    def getProteinWithSiteAndName(self, originalFileName, objectFileName):
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

    def extractWindowSizeSequence(self, sequence, index, windowSize):
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

    def getPartSequence(self, fastaFileName, windowSize, fileType):
        '''
        will get two files PosSequence and NegSequence where each line is a sequence part.
        :param fastaFileName: form is with 01 and like "Trainset.txt"
        :param windowSize:
        :param fileType:Train or Test
        :return:
        '''
        f = open(fastaFileName, "r")
        fPos = open(fileType+"PosSequence.seq", "w", encoding="utf8")
        fNeg = open(fileType+"NegSequence.seq", "w", encoding="utf8")
        i = 0
        while 1:
            line = f.readline()
            if not line:
                break
            sequence = f.readline().strip()
            flags = f.readline().strip()
            for i in range(len(sequence)):
                if sequence[i] == 'K':
                    partSequence = self.extractWindowSizeSequence(sequence=sequence, index=i, windowSize=windowSize)
                    if flags[i] == '1':
                        fPos.write(partSequence+"\n")
                    elif flags[i] == '0':
                        fNeg.write(partSequence+"\n")
        print("separate sequence is okay")

def main():
    # file = "fortrain40.txt"
    # InitFeature().getPartSequence(file, windowSize=8, fileType='Train40KNN')
    fileType = "Train80"
    a,b,c = InitFeature().getTrainData(fileType+"PosSequence.seq", fileType+"NegSequence.seq", "Train80AAIndexPosSequence.seq", "Train80AAIndexNegSequence.seq")
    print(a[100])
    print(b[100])
    # sequence="MTTQQIDLQGPGPWGFRLVGGKDFEQPLAISRVTPGSKAALANLCIGDVITAIDGENTSNMTHLEAQNRIKGCTDNLTLTVARSEHKVWSPLVTEEGKRHPYKMNLASEPQEVLHIGSAHNRSAMPFTASPASSTTARVITNQYNNPAGLYSSENISNFNNALESKTAASGVEANSRPLDHAQPPSSLVIDKESEVYKMLQEKQELNEPPKQSTSFLVLQEILESEEKGDPNKPSGFRSVKAPVTKVAASIGNAQKLPMCDKCGTGIVGVFVKLRDRHRHPECYVCTDCGTNLKQKGHFFVEDQIYCEKHARERVTPPEGYEVVTVFPK"
    # for i in range(len(sequence)):
    #     if sequence[i] == 'K':
    #         partSequence = InitFeature().extractWindowSizeSequence(sequence=sequence, index=i, windowSize=10)
    #         print(partSequence)


if __name__ == '__main__':
    main()