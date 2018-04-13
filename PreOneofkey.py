import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import random
import keras.utils.np_utils as kutils
from tools import *
# from InitFeature import InitFeature


class PreOneOfKey(object):

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


    def preOneofkey(self, posPart, negPart):
        '''
        return two-dimension list object of positive part and negative part with their label information
        :param posPart:
        :param negPart:
        :return:
        '''
        pre_oneofkey_pos = []
        pre_oneofkey_neg = []
        for pp in posPart:
            dividedList = list(pp)
            dividedList.insert(0, '1')
            pre_oneofkey_pos.append(dividedList)
        for np in negPart:
            dividedList = list(np)
            dividedList.insert(0, '0')
            pre_oneofkey_neg.append(dividedList)
        return pre_oneofkey_pos, pre_oneofkey_neg


    def shuffleOneofkeyRespectivly(self, pre_oneofkey_pos, pre_oneofkey_neg, *random_seed):
        """
        shuffle one hot key by there label, positive or negative.
        :param pre_oneofkey_pos:
        :param pre_oneofkey_neg:
        :param random_seed:
        :return:
        """
        if random_seed != ():
            random.seed(random_seed)
        indexP = list(range(0, len(pre_oneofkey_pos)))
        indexN = list(range(0, len(pre_oneofkey_neg)))
        rsp = random.sample(indexP, len(indexP))
        rsn = random.sample(indexN, len(indexN))
        pre_oneofkey_pos = pd.DataFrame(pre_oneofkey_pos).as_matrix()
        pre_oneofkey_neg = pd.DataFrame(pre_oneofkey_neg).as_matrix()
        oneofkey_pos = np.array(pre_oneofkey_pos)[rsp]
        oneofkey_neg = np.array(pre_oneofkey_neg)[rsn]
        return oneofkey_pos, oneofkey_neg


    def convertOneofKeyXY(self, oneofkeyXYData):
        sampleSeq3DArr = oneofkeyXYData[:, 1:]
        oneofkey_x = self.convertSampleToProbMatr(sampleSeq3DArr)
        oneofkey_y = oneofkeyXYData[:, 0]
        oneofkey_y = kutils.to_categorical(oneofkey_y)
        return oneofkey_x, oneofkey_y


    def convertSampleToProbMatr(self, sampleSeq3DArr):
        """
        PARAMETER
        ---------
        sampleSeq3DArr: 2D numpy array
           X denoted the unknow amino acid.
        probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
        """
        letterDict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5,
                      'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                      'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
                      'V': 17, 'W': 18, 'Y': 19, 'X': 0}
        AACategoryLen = 21
        probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
        sampleNo = 0
        for sequence in sampleSeq3DArr:
            AANo = 0
            for AA in sequence:
                if not AA in letterDict:
                    probMatr[sampleNo][0][AANo] = np.full((1, AACategoryLen), 1.0 / AACategoryLen)
                else:
                    index = letterDict[AA]
                    probMatr[sampleNo][0][AANo][index] = 1
                AANo += 1
            sampleNo += 1
        del sampleSeq3DArr
        return probMatr


    def getTrainOneofkeyNLabel(self, positiveFile, negativeFile, selectedPercent=1, random_state=0):
        temp_ook_pos, temp_ook_neg = self.getPosandNegPartFromFiles(positiveFile, negativeFile)
        ook_pos_with_label, ook_neg_with_label = self.preOneofkey(temp_ook_pos, temp_ook_neg)
        ook_pos_with_label, ook_neg_with_label = self.shuffleOneofkeyRespectivly(ook_pos_with_label,
                                                                            ook_neg_with_label, random_state)
        # shuffle every samples
        selectNumber = int(len(ook_pos_with_label) * selectedPercent)
        ook_pos_with_label = ook_pos_with_label[0:selectNumber]
        ook_neg_with_label = ook_neg_with_label[0:selectNumber]
        ook_with_label = np.concatenate((ook_pos_with_label, ook_neg_with_label))
        ook_with_label = shuffleArray(ook_with_label, 0)
        ook_x, ook_y = self.convertOneofKeyXY(ook_with_label)
        return ook_x, ook_y

    def getTestOneofkeyNLabel(self, positiveFile, negativeFile, random_state=0, selectedPercent=1, selectedPecent=0.8):
        temp_ook_pos, temp_ook_neg = self.getPosandNegPartFromFiles(positiveFile, negativeFile)
        ook_pos_with_label, ook_neg_with_label = self.preOneofkey(temp_ook_pos, temp_ook_neg)
        ook_pos_with_label, ook_neg_with_label = self.shuffleOneofkeyRespectivly(ook_pos_with_label,
                                                                            ook_neg_with_label, random_state)
        # shuffle every samples
        ook_with_label = np.concatenate((ook_pos_with_label, ook_neg_with_label))
        # ook_with_label = shuffleArray(ook_with_label, 0)
        ook_x, ook_y = self.convertOneofKeyXY(ook_with_label)
        return ook_x, ook_y


if __name__ == '__main__':
    fastaDir = "fasta/"
    fastaFileName = "fortrain40.fasta"
    Train_or_Test = "Train"
    windowType = "OOK"
    windowSize = 7
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/"+Train_or_Test+windowType+"PosSequence.seq"
    negativeFile = "seq/"+Train_or_Test+windowType+"NegSequence.seq"
    PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1, random_state=0)

    fastaFileName = "TestSet.fasta"
    Train_or_Test = "Test"
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    PreOneOfKey().getTestOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1, random_state=0)
    # getPartSequence(fastaFileName="TrainSet.txt", windowSize=20, fileType="Train")
    # getPartSequence(fastaFileName="TestSet.txt", windowSize=20, fileType="Test")
    # a, b = PreOneOfKey().getTrainOneofkeyNLabel("TrainposSequence.txt", "TrainnegSequence.txt", selectedPecent=1)
    # print(b[0:1000, 1])
    # c, d = getOneofkeyNLabel("TestposSequence.txt", "TestnegSequence.txt")
    # temp_ook_pos, temp_ook_neg = getPosandNegPartFromFiles("posSequence.txt", "negSequence.txt")
    # train_ook_pos_with_label, train_ook_neg_with_label = preOneofkey(temp_ook_pos, temp_ook_neg)
    # train_ook_pos_with_label, train_ook_neg_with_label = shuffleOneofkeyRespectivly(train_ook_pos_with_label,
    #                                                                                 train_ook_neg_with_label, 0)
    #
    # train_ook_x, train_ook_y = convertOneofKeyXY(train_ook_pos_with_label)
    # temp_ook_pos, temp_ook_neg = getPosandNegPartFromFiles("TestposSequence.txt", "TestnegSequence.txt")
    # test_ook_pos_with_label, test_ook_neg_with_label = preOneofkey(temp_ook_pos, temp_ook_neg)
    # test_ook_pos_with_label, test_ook_neg_with_label = shuffleOneofkeyRespectivly(test_ook_pos_with_label,
    #                                                                               test_ook_neg_with_label, 0)
    # test_ook_with_label = np.concatenate((test_ook_pos_with_label, test_ook_neg_with_label))
    # test_ook_with_label = shuffleArray(test_ook_with_label)
    # test_ook_x, test_ook_y = convertOneofKeyXY(test_ook_with_label)

