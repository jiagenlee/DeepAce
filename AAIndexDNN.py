from keras.layers import Input, Flatten, BatchNormalization, Dense, Dropout, AlphaDropout
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.models import Model
from keras.layers.noise import GaussianNoise
import keras
import numpy as np
from Evaluator import Evaluator
from InitAAindex import InitAAindex
from tools import *
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from sklearn.utils.class_weight import compute_class_weight


def aaIndexDNN(trainAAIndexX, trainAAIndexY, nb_epoch=1000, earlystop=None, compile=True, compileModel=None,
               class_weight=None, predict=False, batch_size=2048, verbose=1):
    if class_weight is None:
        class_weight = {0: 0.5, 1: 0.5}
    train_aaindex_X_t = trainAAIndexX
    row = trainAAIndexX.shape[2]
    col = trainAAIndexX.shape[3]
    train_aaindex_X_t.shape = (trainAAIndexX.shape[0], row, col)
    input = Input(shape=(row, col))
    # print("row: ", row)
    # print("col: ", col)
    hidden_number = int(np.sqrt(64*row*col))
    # print("hidden_number: ", hidden_number)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=earlystop)
    nb_epoch = nb_epoch
    if compile == True:
        nb_classes = 2
        batch_size = batch_size
        optimization = 'Nadam'
        aaindex_x = core.Flatten()(input)
        aaindex_x = BatchNormalization()(aaindex_x)
        # attention_probs = Dense(row * col, activation='softmax', name='5')(aaindex_x)
        # aaindex_x = Multiply()([aaindex_x, attention_probs])
        aaindex_x = Dense(256, kernel_initializer='he_uniform', activation='relu', name='111')(aaindex_x)
        aaindex_x = Dropout(0.6)(aaindex_x)
        aaindex_x = Dense(128, kernel_initializer='he_uniform', activation='softplus', name='11')(aaindex_x)
        # aaindex_x = BatchNormalization()(aaindex_x)
        aaindex_x = Dropout(0.55)(aaindex_x)

        aaindex_x = GaussianNoise(10)(aaindex_x)

        output = Dense(64, kernel_initializer='glorot_normal', activation='relu', name='10')(aaindex_x)

        out = Dense(nb_classes, kernel_initializer='glorot_normal', activation='softmax',
                    kernel_regularizer=l2(0.001), name='19')(output)
        aaindex_dnn = Model(input, out)
        aaindex_dnn.compile(loss='binary_crossentropy', optimizer=optimization, metrics=[keras.metrics.binary_accuracy])

    else:
        aaindex_dnn = compileModel

    AAIndex_class_weight = class_weight

    weight_checkpointer = ModelCheckpoint(filepath='AAIndex.h5', verbose=verbose, save_best_only=True,
                                            monitor='val_binary_accuracy', mode='max', save_weights_only=True)

    aaindex_dnn.fit(train_aaindex_X_t, trainAAIndexY, batch_size=batch_size, epochs=nb_epoch,
                    shuffle=True, validation_split=0.2, class_weight=AAIndex_class_weight,
                    callbacks=[early_stopping, weight_checkpointer], verbose=verbose)
    return aaindex_dnn

def main():
    model = None
    fastaDir = "fasta/"
    fastaFileName = "fortrain40.fasta"
    Train_or_Test = "Train"
    windowSize = 14
    windowType = "AAIndex"
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    try:
        trainX, trainY = InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=0.8, random_state=0)
    except MemoryError as au:
        print("Window Size: ", windowSize, " ", au)

    model = aaIndexDNN(trainX, trainY, nb_epoch=1000, earlystop=20, compile=True, compileModel=None,
                           class_weight={0: 0.6, 1: 0.4}, predict=False, batch_size=2048, verbose=0)

    fastaFileName = "TestSet.fasta"
    Train_or_Test = "Test"
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    testX, testY = InitAAindex().getTestAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)
    testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])
    # score, acc = model.evaluate(testX, testY)
    # return {'loss': -acc, 'status': STATUS_OK, 'model':model}
    predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
    y_pred = predict_probability.argmax(axis=-1)
    result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
    mcc = -1.0
    for rs in np.random.randint(10, 10000, 10):
        Train_or_Test = "Train"
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        trainX, trainY = InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=1,
                                                       random_state=rs)
        model = aaIndexDNN(trainX, trainY, nb_epoch=1000, earlystop=20, compile=True, compileModel=None,
                           class_weight={0: 0.6, 1: 0.4}, predict=False, batch_size=2048, verbose=0)
        predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
        y_pred = predict_probability.argmax(axis=-1)
        result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
        if result[9] > mcc:
            mcc = result[9]
            model.save("model/aaindex_" + str(mcc) + ".h5")


def testWindowSize():
    for windowSize in range(5, 31):
        model = None
        print(windowSize)
        totalResult = []
        fastaDir = "fasta/"
        fastaFileName = "fortrain40.fasta"
        Train_or_Test = "Train"
        windowType = "AAIndex"
        getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        try:
            trainX, trainY = InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)
        except MemoryError as au:
            print("Window Size: ", windowSize, " ", au)
        else:
            model = aaIndexDNN(trainX, trainY, nb_epoch=1000, earlystop=20, compile=True, compileModel=None,
                               class_weight={0: 0.52, 1: 0.48}, predict=False, batch_size=2048, verbose=0)

        fastaFileName = "TestSet.fasta"
        Train_or_Test = "Test"
        getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        testX, testY = InitAAindex().getTestAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)
        testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])
        predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
        y_pred = predict_probability.argmax(axis=-1)
        result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
        totalResult.append(result)
        for rs in np.random.randint(10, 10000, 10):
            Train_or_Test = "Train"
            positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
            negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
            trainX, trainY = InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=1,
                                                           random_state=rs)
            model = aaIndexDNN(trainX, trainY, nb_epoch=1000, earlystop=20, compile=True, compileModel=None,
                               class_weight={0: 0.52, 1: 0.48}, predict=False, batch_size=2048, verbose=0)
            predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
            y_pred = predict_probability.argmax(axis=-1)
            result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
            totalResult.append(result)
        # print(totalResult)


if __name__ == '__main__':
    main()
    # testWindowSize()