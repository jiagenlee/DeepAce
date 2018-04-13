import numpy as np
# np.set_printoptions(threshold=np.nan)
import pandas as pd
import random
import keras.utils.np_utils as kutils
# from InitFeature import InitFeature
from tools import *


class InitAAindex(object):

    def getPosandNegPartFromFiles(self, positiveFile, negativeFile):
        '''
        based on two part seqence file
        files should be used to store part of sequence information.
        positive fragment and negative should be putted respectively into files .
        :param positiveFile:
        :param negativeFile:
        :return:
        '''
        fp = open(positiveFile, "r")
        fn = open(negativeFile, "r")
        posPart = []
        negPart = []
        for line in fp:
            posPart.append(line.strip())
        fp.close()
        for line in fn:
            negPart.append(line.strip())
        fn.close()
        return posPart, negPart

    def preAAIndex(self, posPart, negPart):
        '''
        return two-dimension list object of positive part and negative part with their label information
        :param posPart:
        :param negPart:
        :return: pre_aaindex_pos, pre_aaindex_neg
        '''
        pre_aaindex_pos = []
        pre_aaindex_neg = []
        for pp in posPart:
            dividedList = list(pp)
            dividedList.insert(0, '1')
            pre_aaindex_pos.append(dividedList)
        for np in negPart:
            dividedList = list(np)
            dividedList.insert(0, '0')
            pre_aaindex_neg.append(dividedList)
        return pre_aaindex_pos, pre_aaindex_neg

    def shuffleOneofkeyRespectivly(self, pre_aaindex_pos, pre_aaindex_neg, *random_seed):
        """
        shuffle one hot key by there label, positive or negative.
        :param pre_aaindex_pos:
        :param pre_aaindex_neg:
        :param random_seed:
        :return: aaindex_pos and aaindex_neg belong to numpy.ndarray with their flag
        """
        if random_seed != ():
            random.seed(random_seed)
        indexP = list(range(0, len(pre_aaindex_pos)))
        indexN = list(range(0, len(pre_aaindex_neg)))
        rsp = random.sample(indexP, len(indexP))
        rsn = random.sample(indexN, len(indexN))
        pre_aaindex_pos = pd.DataFrame(pre_aaindex_pos).as_matrix()
        pre_aaindex_neg = pd.DataFrame(pre_aaindex_neg).as_matrix()
        aaindex_pos = np.array(pre_aaindex_pos)[rsp]
        aaindex_neg = np.array(pre_aaindex_neg)[rsn]
        return aaindex_pos, aaindex_neg

    def convertAAIndexXY(self, aaindexData):
        sampleSeq3DArr = aaindexData[:, 1:]
        # aaindex_x = self.convertSampleToAAIndexFactors5(sampleSeq3DArr)
        aaindex_x = self.convertSampleToAAIndex28(sampleSeq3DArr)
        # aaindex_x = self.convertSampleToAllAAIndex(sampleSeq3DArr)
        aaindex_y = aaindexData[:, 0]
        aaindex_y = kutils.to_categorical(aaindex_y)
        return aaindex_x, aaindex_y


    def convertSampleToAAIndexFactors5(self, sampleSeq3DArr):
        """
        Convertd the raw data to 5 factors that represent AAindex

        PARAMETER
        ---------
        sampleSeq3DArr: 3D numpy array
            X denoted the unknow amino acid.
        probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
        """
        letterDict = {}
        letterDict["A"] = [-0.591, -1.302, -0.733, 1.570, -0.146]
        letterDict["C"] = [-1.343, 0.465, -0.862, -1.020, -0.255]
        letterDict["D"] = [1.050, 0.302, -3.656, -0.259, -3.242]
        letterDict["E"] = [1.357, -1.453, 1.477, 0.113, -0.837]
        letterDict["F"] = [-1.006, -0.590, 1.891, -0.397, 0.412]
        letterDict["G"] = [-0.384, 1.652, 1.330, 1.045, 2.064]
        letterDict["H"] = [0.336, -0.417, -1.673, -1.474, -0.078]
        letterDict["I"] = [-1.239, -0.547, 2.131, 0.393, 0.816]
        letterDict["K"] = [1.831, -0.561, 0.533, -0.277, 1.648]
        letterDict["L"] = [-1.019, -0.987, -1.505, 1.266, -0.912]
        letterDict["M"] = [-0.663, -1.524, 2.219, -1.005, 1.212]
        letterDict["N"] = [0.945, 0.828, 1.299, -0.169, 0.933]
        letterDict["P"] = [0.189, 2.081, -1.628, 0.421, -1.392]
        letterDict["Q"] = [0.931, -0.179, -3.005, -0.503, -1.853]
        letterDict["R"] = [1.538, -0.055, 1.502, 0.440, 2.897]
        letterDict["S"] = [-0.228, 1.399, -4.760, 0.670, -2.647]
        letterDict["T"] = [-0.032, 0.326, 2.213, 0.908, 1.313]
        letterDict["V"] = [-1.337, -0.279, -0.544, 1.242, -1.262]
        letterDict["W"] = [-0.595, 0.009, 0.672, -2.128, -0.184]
        letterDict["Y"] = [0.260, 0.830, 3.097, -0.838, 1.512]
        AACategoryLen = 5
        aaIndexFactors5 = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
        sampleNo = 0
        for sequence in sampleSeq3DArr:
            AANo = 0
            for AA in sequence:
                if not AA in letterDict:
                    aaIndexFactors5[sampleNo][0][AANo] = np.full((1, AACategoryLen), 0)
                else:
                    aaIndexFactors5[sampleNo][0][AANo] = letterDict[AA]
                AANo += 1
            sampleNo += 1
        return aaIndexFactors5

    def convertSampleToAAIndex28(self, sampleSeq3DArr):
        """
        Convertd the raw data to 5 factors that represent AAindex

        PARAMETER
        ---------
        sampleSeq3DArr: 3D numpy array
            X denoted the unknow amino acid.
        probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
        """
        # aaIndex = np.loadtxt("fasta/AAIndexValue.txt", dtype=np.float, delimiter=" ")
        aaIndex = np.loadtxt("fasta/new_feature_value_3.txt", dtype=np.float, delimiter=" ")
        # sspka_aaindex = np.loadtxt("fasta/np_sspka_aaindex.txt", dtype=np.float, delimiter=" ")
        # aaIndex = np.concatenate((aaIndex, sspka_aaindex), axis=1)
        letterDict = {}
        letter = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        for i in range(len(letter)):
            letterDict[letter[i]] = aaIndex[i]
        # letterDict = {}
        # letterDict["A"] = aaIndex[0]
        # letterDict["C"] = aaIndex[1]
        # letterDict["D"] = aaIndex[2]
        # letterDict["E"] = aaIndex[3]
        # letterDict["F"] = aaIndex[4]
        # letterDict["G"] = aaIndex[5]
        # letterDict["H"] = aaIndex[6]
        # letterDict["I"] = aaIndex[7]
        # letterDict["K"] = aaIndex[8]
        # letterDict["L"] = aaIndex[9]
        # letterDict["M"] = aaIndex[10]
        # letterDict["N"] = aaIndex[11]
        # letterDict["P"] = aaIndex[12]
        # letterDict["Q"] = aaIndex[13]
        # letterDict["R"] = aaIndex[14]
        # letterDict["S"] = aaIndex[15]
        # letterDict["T"] = aaIndex[16]
        # letterDict["V"] = aaIndex[17]
        # letterDict["W"] = aaIndex[18]
        # letterDict["Y"] = aaIndex[19]
        AACategoryLen = len(aaIndex[0])
        aaIndexFactors5 = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
        sampleNo = 0
        for sequence in sampleSeq3DArr:
            AANo = 0
            for AA in sequence:
                if not AA in letterDict:
                    aaIndexFactors5[sampleNo][0][AANo] = np.full((1, AACategoryLen), 0)
                else:
                    aaIndexFactors5[sampleNo][0][AANo] = letterDict[AA]
                AANo += 1
            sampleNo += 1
        return aaIndexFactors5

    def convertSampleToAllAAIndex(self, sampleSeq3DArr):
        """
        Convertd the raw data to 5 factors that represent AAindex

        PARAMETER
        ---------
        sampleSeq3DArr: 3D numpy array
            X denoted the unknow amino acid.
        probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
        """
        aaIndex = np.loadtxt("aaindexAll.txt", dtype=np.float, delimiter=",")
        letterDict = {}
        letterDict["A"] = aaIndex[0]
        letterDict["C"] = aaIndex[1]
        letterDict["D"] = aaIndex[2]
        letterDict["E"] = aaIndex[3]
        letterDict["F"] = aaIndex[4]
        letterDict["G"] = aaIndex[5]
        letterDict["H"] = aaIndex[6]
        letterDict["I"] = aaIndex[7]
        letterDict["K"] = aaIndex[8]
        letterDict["L"] = aaIndex[9]
        letterDict["M"] = aaIndex[10]
        letterDict["N"] = aaIndex[11]
        letterDict["P"] = aaIndex[12]
        letterDict["Q"] = aaIndex[13]
        letterDict["R"] = aaIndex[14]
        letterDict["S"] = aaIndex[15]
        letterDict["T"] = aaIndex[16]
        letterDict["V"] = aaIndex[17]
        letterDict["W"] = aaIndex[18]
        letterDict["Y"] = aaIndex[19]
        AACategoryLen = len(aaIndex[0])
        aaIndexFactors5 = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
        sampleNo = 0
        for sequence in sampleSeq3DArr:
            AANo = 0
            for AA in sequence:
                if not AA in letterDict:
                    aaIndexFactors5[sampleNo][0][AANo] = np.full((1, AACategoryLen), 0)
                else:
                    aaIndexFactors5[sampleNo][0][AANo] = letterDict[AA]
                AANo += 1
            sampleNo += 1
        return aaIndexFactors5

    def getTrainAAIndex(self, positiveFile, negativeFile, selectedPercent=1, random_state=0):
        posPart, negPart = self.getPosandNegPartFromFiles(positiveFile, negativeFile)
        pre_aaindex_pos, pre_aaindex_neg = self.preAAIndex(posPart, negPart)
        aaindex_pos_with_label, aaindex_neg_with_label = self.shuffleOneofkeyRespectivly(pre_aaindex_pos, pre_aaindex_neg, random_state)
        # aaindexData = np.concatenate((aaindex_pos_with_label, aaindex_neg_with_label))
        selectPosNumber = int(len(aaindex_pos_with_label) * selectedPercent)
        selectNegNumber = int(len(aaindex_pos_with_label) * selectedPercent)
        aaindex_pos_with_label = aaindex_pos_with_label[0:selectPosNumber]
        aaindex_neg_with_label = aaindex_neg_with_label[0:selectNegNumber]
        aaindex_with_label = np.concatenate((aaindex_pos_with_label, aaindex_neg_with_label))
        aaindex_with_label = shuffleArray(aaindex_with_label, 0)
        aaindex_x, aaindex_y = self.convertAAIndexXY(aaindex_with_label)
        return aaindex_x, aaindex_y

    def getTestAAIndex(self, positiveFile, negativeFile, selectedPercent=1, random_state=0):
        posPart, negPart = self.getPosandNegPartFromFiles(positiveFile, negativeFile)
        pre_aaindex_pos, pre_aaindex_neg = self.preAAIndex(posPart, negPart)
        aaindex_pos, aaindex_neg = self.shuffleOneofkeyRespectivly(pre_aaindex_pos, pre_aaindex_neg, random_state)
        aaindexData = np.concatenate((aaindex_pos, aaindex_neg))
        aaindex_pos_with_label = aaindex_pos[0:]
        aaindex_neg_with_label = aaindex_neg[0:]
        aaindex_with_label = np.concatenate((aaindex_pos_with_label, aaindex_neg_with_label))
        # aaindex_with_label = shuffleArray(aaindex_with_label, 0)
        aaindex_x, aaindex_y = self.convertAAIndexXY(aaindex_with_label)
        return aaindex_x, aaindex_y


if __name__ == '__main__':
    fastaDir = "fasta/"
    fastaFileName = "fortrain40.fasta"
    Train_or_Test = "Train"
    windowType = "AAIndex"
    windowSize = 7
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)

    fastaFileName = "TestSet.fasta"
    Train_or_Test = "Test"
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    InitAAindex().getTestAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)
    # a, b = ia.getAAIndex("Train80PosSequence.seq", "Train80NegSequence.seq")