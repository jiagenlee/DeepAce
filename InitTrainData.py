import random
import numpy as np
import pandas as pd
from PreOneofkey import PreOneOfKey
from InitAAindex import InitAAindex
from tools import shuffleTwoArray
import pickle


class InitTrainData(object):
    def shuffleDataRespectivly(self, pre_oneofkey_pos, pre_oneofkey_neg,
                               pre_aaindex_pos, pre_aaindex_neg, *random_seed):
        """
        shuffle by there label, positive and negative for each.
        :param pre_oneofkey_pos:
        :param pre_oneofkey_neg:
        :param pre_aaindex_pos:
        :param pre_aaindex_neg
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
        pre_aaindex_pos = pd.DataFrame(pre_aaindex_pos).as_matrix()
        pre_aaindex_neg = pd.DataFrame(pre_aaindex_neg).as_matrix()
        oneofkey_pos = np.array(pre_oneofkey_pos)[rsp]
        oneofkey_neg = np.array(pre_oneofkey_neg)[rsn]
        aaindex_pos = np.array(pre_aaindex_pos)[rsp]
        aaindex_neg = np.array(pre_aaindex_neg)[rsn]
        return oneofkey_pos, oneofkey_neg, aaindex_pos, aaindex_neg

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

    def getTrainData(self, positiveOokFile, negativeOokFile,
                     positiveAAIndexFile, negativeAAIndexFile,
                     selectedPercent=1, randam_state=0):
        '''
        for megerd network. this is the train data.
        :param positiveOokFile:
        :param negativeOokFile:
        :param positiveAAIndexFile:
        :param NegativeAAIndexFile:
        :param selectedPercent:
        :param randam_state:
        :return:
        '''
        posOokPart, negOokPart = self.getPosandNegPartFromFiles(positiveOokFile, negativeOokFile)
        posAAIndexPart, negAAIndexPart = self.getPosandNegPartFromFiles(positiveAAIndexFile, negativeAAIndexFile)
        po = PreOneOfKey()
        ia = InitAAindex()
        temp_ook_pos, temp_ook_neg = po.preOneofkey(posOokPart, negOokPart)
        temp_index_pos, temp_index_neg = ia.preAAIndex(posAAIndexPart, negAAIndexPart)
        ook_pos_with_label, ook_neg_with_label, aaindex_pos_with_label, aaindex_neg_with_label = self.shuffleDataRespectivly(temp_ook_pos, temp_ook_neg, temp_index_pos, temp_index_neg)
        selectNumber = int(len(aaindex_pos_with_label)*selectedPercent)
        ook_pos_with_label = ook_pos_with_label[0:selectNumber]
        ook_neg_with_label = ook_neg_with_label[0:selectNumber]
        aaindex_pos_with_label = aaindex_pos_with_label[0:selectNumber]
        aaindex_neg_with_label = aaindex_neg_with_label[0:selectNumber]
        ook_with_label = np.concatenate((ook_pos_with_label, ook_neg_with_label))
        aaindex_with_label = np.concatenate((aaindex_pos_with_label, aaindex_neg_with_label))
        ook_with_label, aaindex_with_label = shuffleTwoArray((ook_with_label, aaindex_with_label), 0)
        ook_x, ook_y = po.convertOneofKeyXY(ook_with_label)
        aaindex_x, aaindex_y = ia.convertAAIndexXY(aaindex_with_label)
        return ook_x, aaindex_x, ook_y

    def getTestData(self, positiveOokFile, negativeOokFile,
                    positiveAAIndexFile, negativeAAIndexFile, selectedPercent=1, randam_state=0):
        posOokPart, negOokPart = self.getPosandNegPartFromFiles(positiveOokFile, negativeOokFile)
        posAAIndexPart, negAAIndexPart = self.getPosandNegPartFromFiles(positiveAAIndexFile, negativeAAIndexFile)
        po = PreOneOfKey()
        ia = InitAAindex()
        temp_ook_pos, temp_ook_neg = po.preOneofkey(posOokPart, negOokPart)
        temp_index_pos, temp_index_neg = ia.preAAIndex(posAAIndexPart, negAAIndexPart)
        ook_pos_with_label, ook_neg_with_label, aaindex_pos_with_label, aaindex_neg_with_label = self.shuffleDataRespectivly(temp_ook_pos, temp_ook_neg, temp_index_pos, temp_index_neg)
        ook_with_label = np.concatenate((ook_pos_with_label, ook_neg_with_label))
        aaindex_with_label = np.concatenate((aaindex_pos_with_label, aaindex_neg_with_label))
        # ook_with_label, aaindex_with_label = shuffleTwoArray((ook_with_label, aaindex_with_label), 0)
        ook_x, ook_y = po.convertOneofKeyXY(ook_with_label)
        aaindex_x, aaindex_y = ia.convertAAIndexXY(aaindex_with_label)
        # pickle.dump(ook_x, open("TestOOkX.pickle","w"))
        # pickle.dump(aaindex_x, open("TestAAindexX.pickle","w"))
        # pickle.dump(ook_y, open("TestBothY.pickle","w"))
        np.save("TestOOkX", ook_x, allow_pickle=True)
        np.save("TestAAindexX", aaindex_x, allow_pickle=True)
        np.save("TestBothY", ook_y, allow_pickle=True)
        print("succeed")
        # return ook_x, aaindex_x, ook_y

    def loadTestDataFromNpy(self, ookNPy, aaindexNpy, yNpy):
        ook_x = np.load(ookNPy)
        aaindex_x = np.load(aaindexNpy)
        ook_y = np.load(yNpy)
        return ook_x, aaindex_x, ook_y


def main():
    # file = "fortrain40.txt"
    # InitFeature().getPartSequence(file, windowSize=8, fileType='Train40KNN')
    fileType = "Train80"
    a, b, c = InitTrainData().getTrainData(fileType + "PosSequence.seq", fileType + "NegSequence.seq", "Train80AAIndexPosSequence.seq", "Train80AAIndexNegSequence.seq")
    # InitTrainData().getTestData("TestposSequence.txt", "TestnegSequence.txt", "TestAAIndexPosSequence.seq", "TestAAIndexNegSequence.seq")
    del a, b, c
    d, e, f = InitTrainData().loadTestDataFromNpy("TestOOkX.npy", "TestAAindexX.npy", "TestBothY.npy")
    print(d[20])
    print(e[20])
    print(f[20])


if __name__ == '__main__':
    main()