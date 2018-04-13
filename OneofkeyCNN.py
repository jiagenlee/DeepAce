# from PreOneofkey import getTrainOneofkeyNLabel, getTestOneofkeyNLabel
from PreOneofkey import PreOneOfKey
import keras.layers.core as core
import keras.layers.convolutional as conv
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
import keras.metrics
from Evaluator import Evaluator
from InitFeature import InitFeature
import numpy as np
from tools import *
from keras.models import load_model
from Attention import Attention, myFlatten


def OOKCNN(trainX, trainY, nb_epoch, earlystop=None, compiletimes=0, compilemodels=None,
             batch_size=2048, class_weights={0: 1, 1: 1}, predict=False):
    #Set Oneofkey Network Size and Data
    input_row = trainX.shape[2]
    input_col = trainX.shape[3]
    trainX_t = trainX
    # Early_stop
    if (earlystop is not None):
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=earlystop)
    # set to a very big value since earlystop used
    nb_epoch = nb_epoch
    # TrainX_t For Shape
    trainX_t.shape = (trainX_t.shape[0], input_row, input_col)
    input = Input(shape=(input_row, input_col))

    # params = {'dropout1': 0.09055921027754717, 'dropout2': 0.6391239298866936, 'dropout3': 0.4494981811340072,
    #  'dropout4': 0.13858850326177857, 'dropout5': 0.37168935754516325, 'layer1_node': 21.380001953812567,
    #  'layer1_size': 1, 'layer2_node': 42.3937544103545, 'layer2_size': 16, 'layer3_node': 184.87943202539697,
    #  'layer4_node': 61.85302597240724, 'layer5_node': 415.9952475249118, 'nb_epoch': 178, 'windowSize': 16}

    # layer1_node = int(params["layer1_node"])
    # layer2_node = int(params["layer2_node"])
    # layer3_node = int(params["layer3_node"])
    # layer4_node = int(params["layer4_node"])
    # layer5_node = int(params["layer5_node"])
    # layer1_size = params["layer1_size"]
    # layer2_size = params["layer2_size"]
    # dropout1 = params["dropout1"]
    # dropout2 = params["dropout2"]
    # dropout3 = params["dropout3"]
    # dropout4 = params["dropout4"]
    # dropout5 = params["dropout5"]

    if compiletimes == 0:
        # Total Set Classes
        nb_classes = 2
        # Total Set Batch_size
        batch_size = 8192
        # Total Set Optimizer
        # optimizer = SGD(lr=0.0001, momentum=0.9, nesterov= True)
        optimization = 'Nadam'
        #begin of Oneofkey Network

        # x = conv.Conv1D(layer1_node, layer1_size, name="layer1", kernel_initializer="glorot_normal",
        #                 kernel_regularizer=l2(0), padding="same")(input)
        # x = Dropout(dropout1)(input)
        # x = Activation('softsign')(x)
        #
        # x = conv.Conv1D(layer2_node, layer2_size, name="layer2", kernel_initializer="glorot_normal",
        #                 kernel_regularizer=l2(0), padding="same")(x)
        # x = Dropout(dropout2)(x)
        # x = Activation('softsign')(x)
        #
        # output_x = core.Flatten()(x)
        # output = BatchNormalization()(output_x)
        # output = Dropout(dropout3)(output)
        #
        # # attention_probs = Dense(1155, activation='softmax', name='attention_probs')(output)
        # # attention_mul = Multiply()([output, attention_probs])
        #
        # output = Dense(layer3_node, kernel_initializer='glorot_normal', activation='relu', name='layer3')(output)
        # output = Dropout(dropout4)(output)
        # output = Dense(layer4_node, kernel_initializer='glorot_normal', activation="relu", name='layer4')(output)
        # output = Dropout(dropout5)(output)
        # output = Dense(layer5_node, kernel_initializer='glorot_normal', activation="relu", name='layer5')(output)
        # End of Oneofkey Network
        # out = Dense(nb_classes, kernel_initializer='glorot_normal', activation='softmax', kernel_regularizer=l2(0.001),
        #             name='7')(output)
        #
        # cnn = Model(input, out)
        # cnn.compile(loss=keras.losses.binary_crossentropy, optimizer=optimization, metrics=[keras.metrics.binary_accuracy])
        x = conv.Conv1D(51, 2, name="0", kernel_initializer="glorot_normal", kernel_regularizer=l2(0), padding="same")(
            input)
        x = Dropout(0.3)(x)
        x = Activation('softsign')(x)

        x = conv.Conv1D(21, 3, name="1", kernel_initializer="glorot_normal", kernel_regularizer=l2(0), padding="same")(
            x)
        x = Dropout(0.4)(x)
        x = Activation('softsign')(x)

        # x = conv.Conv1D(21, 5, name="2", kernel_initializer="glorot_normal", kernel_regularizer=l2(0), padding="same")(x)
        # x = Dropout(0.4)(x)
        # x = Activation('softsign')(x)
        #
        # x = conv.Conv1D(101, 7, name="3", kernel_initializer="glorot_normal", kernel_regularizer=l2(0), padding="same")(x)
        # x = Activation('softsign')(x)
        # # x_reshape = core.Reshape((x._keras_shape[2], x._keras_shape[1]))(x)
        # x = Dropout(0.4)(x)

        output_x = core.Flatten()(x)
        output = BatchNormalization()(output_x)
        output = Dropout(0.3)(output)

        # attention_probs = Dense(1155, activation='softmax', name='attention_probs')(output)
        # attention_mul = Multiply()([output, attention_probs])

        output = Dense(128, kernel_initializer='glorot_normal', activation='relu', name='4')(output)
        output = Dropout(0.2)(output)
        output = Dense(64, kernel_initializer='glorot_normal', activation="relu", name='5')(output)
        output = Dropout(0.2)(output)
        output = Dense(415, kernel_initializer='glorot_normal', activation="relu", name='6')(output)
        # End of Oneofkey Network
        out = Dense(nb_classes, kernel_initializer='glorot_normal', activation='softmax', kernel_regularizer=l2(0.001),
                    name='7')(output)

        cnn = Model(input, out)
        cnn.compile(loss=keras.losses.binary_crossentropy, optimizer=optimization,
                    metrics=[keras.metrics.binary_accuracy])
    else:
        cnn = compilemodels

    oneofkclass_weights = class_weights

    if (predict is False):
        if (trainY is not None):
            if (earlystop is None):
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
            else:
                # checkpointer = ModelCheckpoint(filepath='oneofk.h5', verbose=1, save_best_only=True)
                weight_checkpointer = ModelCheckpoint(filepath='oneofkweight9.h5', verbose=0, save_best_only=True,
                                                      monitor='val_binary_accuracy', mode='max', save_weights_only=True)
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch, shuffle=True,
                                     validation_split=0.2, callbacks=[early_stopping, weight_checkpointer],
                                     class_weight=oneofkclass_weights, verbose=0)
        else:
            fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
    return cnn


def testWindowSize():
    for windowSize in range(5, 31):
        model = None
        print(windowSize)
        totalResult = []
        fastaDir = "fasta/"
        fastaFileName = "fortrain40.fasta"
        Train_or_Test = "Train"
        windowType = "OOK"
        getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        try:
            trainX, trainY = PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1, random_state=0)
        except MemoryError as au:
            print("Window Size: ", windowSize, " ", au)
        else:
            model = OOKCNN(trainX, trainY, nb_epoch=1000, earlystop=20, compiletimes=0, compilemodels=None,
                           batch_size=2048, class_weights={0: 1, 1: 1}, predict=False)

        fastaFileName = "TestSet.fasta"
        Train_or_Test = "Test"
        getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        testX, testY = PreOneOfKey().getTestOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1,
                                                           random_state=0)
        testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])
        predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
        y_pred = predict_probability.argmax(axis=-1)
        result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
        totalResult.append(result)

        for rs in np.random.randint(10, 10000, 10):
            Train_or_Test = "Train"
            positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
            negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
            trainX, trainY = PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1,
                                                                  random_state=rs)
            model = OOKCNN(trainX, trainY, nb_epoch=1000, earlystop=20, compiletimes=1, compilemodels=model,
                           batch_size=2048, class_weights={0: 1, 1: 1}, predict=False)
            predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
            y_pred = predict_probability.argmax(axis=-1)
            result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
            totalResult.append(result)
        print(totalResult)

def main():
    # define a file for result
    model = None
    windowSize = 26
    totalResult = []
    fastaDir = "fasta/"
    fastaFileName = "fortrain40.fasta"
    Train_or_Test = "Train"
    windowType = "OOK"
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    try:
        trainX, trainY = PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1,
                                                              random_state=0)
    except MemoryError as au:
        print("Window Size: ", windowSize, " ", au)
    else:
        model = OOKCNN(trainX, trainY, nb_epoch=1000, earlystop=40, compiletimes=0, compilemodels=None,
                       batch_size=2048, class_weights={0: 1, 1: 1}, predict=False)

    fastaFileName = "TestSet.fasta"
    Train_or_Test = "Test"
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    testX, testY = PreOneOfKey().getTestOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1,
                                                       random_state=0)
    testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])
    predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
    y_pred = predict_probability.argmax(axis=-1)
    result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
    mcc = -1.0
    for rs in np.random.randint(10, 10000, 10):
        Train_or_Test = "Train"
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        trainX, trainY = PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=1,
                                                              random_state=rs)
        model = OOKCNN(trainX, trainY, nb_epoch=1000, earlystop=20, compiletimes=1, compilemodels=model,
                       batch_size=2048, class_weights={0: 0.5, 1: 0.5}, predict=False)
        predict_probability = model.predict(x=testX, batch_size=1000, verbose=0)
        y_pred = predict_probability.argmax(axis=-1)
        result = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
        if result[9] > mcc:
            mcc = result[9]
            model.save("model/ook_" + str(mcc) + ".h5")

def testRedundancy():
    # define a file for result
    f = open('C:/Users/zhaoxy/Desktop/Acetylation_log/80NegRm80redundancy.txt', 'w', encoding="utf8")
    fileType = "Train80"
    makeFastaFileIntoSequence("Train80NegRm80redundancy.fasta")
    # if you want make two sequences between positive and negative de-reduntancy, just insert file below 1st line
    trainX, trainY = PreOneOfKey().getTrainOneofkeyNLabel(fileType+"PosSequence.seq", fileType+"NegRm80redundancy.seq", selectedPercent=0.8)

    testX, testY = PreOneOfKey().getTestOneofkeyNLabel("TestposSequence.txt", "TestnegSequence.txt")
    testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])
    model = OOKCNN(trainX, trainY, nb_epoch=1000, earlystop=20, compiletimes=0, compilemodels=None,
                     batch_size=2048, class_weights={0: 1, 1: 1}, predict=False)
    predict_probability = model.predict(x=testX, batch_size=1000, verbose=1)
    y_pred = predict_probability.argmax(axis=-1)
    ss = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
    print(ss)
    f.write(ss+'\n')
    # train by 10 batch
    for rs in np.random.randint(10, 10000, 20):
        f.write("random seed is: "+str(rs)+"\n")
        trainX, trainY = PreOneOfKey().getTestOneofkeyNLabel(fileType + "PosSequence.seq",
                                                             fileType + "NegRm80redundancy.seq", random_state=rs,
                                                             selectedPercent=0.8)
        model = OOKCNN(trainX, trainY, nb_epoch=1000, earlystop=20, compiletimes=1, compilemodels=model,
                     batch_size=2048, class_weights={0: 1, 1: 1}, predict=False)
        predict_probability = model.predict(x=testX, batch_size=1000, verbose=1)
        y_pred = predict_probability.argmax(axis=-1)
        ss = Evaluator().calculate_performance(testY[:, 1], y_pred, predict_probability[:, 1])
        print(ss)
        f.write(ss + '\n')
    f.close()

def drawRoc():
    model = load_model("model/ook_0.242590783076.h5")
    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=30)

    fastaDir = "fasta/"
    fastaFileName = "TestSet.fasta"
    Train_or_Test = "Test"
    windowType = "OOK"
    windowSize = 26
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    test_ook_x, test_ook_y = PreOneOfKey().getTestOneofkeyNLabel(positiveFile, negativeFile, random_state=0)
    print(test_ook_x.shape)

    test_ook_x.shape = (test_ook_x.shape[0], test_ook_x.shape[2], test_ook_x.shape[3])

    predict_probability = model.predict(x=test_ook_x, batch_size=1000, verbose=0)

    y_pred = predict_probability.argmax(axis=-1)
    # precision, recall, _ = precision_recall_curve(test_ook_y[:, 1], predict_probability[:, 1])
    # print(precision)
    # print(recall)
    result = Evaluator().calculate_performance(test_ook_y[:, 1], y_pred, predict_probability[:, 1])


if __name__ == '__main__':
    # drawRoc()
    # testWindowSize()
    main()
