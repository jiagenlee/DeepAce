import keras.layers.core as core
import keras.layers.convolutional as conv
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, History
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
import keras.metrics
from InitAAindex import InitAAindex
from PreOneofkey import PreOneOfKey
from Evaluator import Evaluator
from tools import *
from InitFeature import InitFeature
import numpy as np
from InitTrainData import InitTrainData
from tools import makeFastafileInto01Fasta, makeFastaFileIntoSequence


def MultiCNN(train_ook_X, trainAAIndexX, train_ook_Y, nb_epoch, earlystop=None, compiletimes=0, batch_size=2048,
             predict=False, compileModel=None, class_weight={0: 0.5, 1: 0.5}, verbose=1, model_id=0):
    # Set Oneofkey Data
    ook_row = train_ook_X.shape[2]
    ook_col = train_ook_X.shape[3]
    ook_x_t = train_ook_X
    ook_x_t.shape = (ook_x_t.shape[0], ook_row, ook_col)
    ook_input = Input(shape=(ook_row, ook_col))
    # AAindex
    aaindex_x_t = trainAAIndexX
    aaindex_row = trainAAIndexX.shape[2]
    aaindex_col = trainAAIndexX.shape[3]
    aaindex_x_t.shape = (trainAAIndexX.shape[0], aaindex_row, aaindex_col)
    aaindex_input = Input(shape=(aaindex_row, aaindex_col))

    if (earlystop is not None):
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=earlystop)

    nb_epoch = nb_epoch
    # TrainX_t For Shape
    if compiletimes == 0:
        # Total Set Classes
        nb_classes = 2
        # Total Set Batch_size
        batch_size = 8192
        # Total Set Optimizer
        # optimizer = SGD(lr=0.0001, momentum=0.9, nesterov= True)
        optimization = 'Nadam'
        # begin of Oneofkey Network
        ook_x = conv.Conv1D(51, 2, name="0", kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(0), padding="same")(ook_input)
        ook_x = Dropout(0.3)(ook_x)
        ook_x = Activation('softsign')(ook_x)

        ook_x = conv.Conv1D(21, 3, name="1", kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(0), padding="same")(ook_x)
        ook_x = Dropout(0.4)(ook_x)
        ook_x = Activation('softsign')(ook_x)

        output_ook_x = core.Flatten()(ook_x)
        output_ook_x = BatchNormalization()(output_ook_x)
        output_ook_x = Dropout(0.3)(output_ook_x)

        output_ook_x = Dense(128, kernel_initializer='glorot_normal', activation='relu', name='2')(output_ook_x)
        output_ook_x = Dropout(0.2)(output_ook_x)
        output_ook_x = Dense(64, kernel_initializer='glorot_normal', activation="relu", name='3')(output_ook_x)
        output_ook_x = Dropout(0.2)(output_ook_x)
        # below modified
        output_ook_x = Dense(415, kernel_initializer='glorot_normal', activation="relu", name='4')(output_ook_x)
        # output_ook_x = Dense(nb_classes, kernel_initializer='glorot_normal', activation='softmax', kernel_regularizer=l2(0.001),
        #             name='7')(output_ook_x)
        # End of Oneofkey Network

        # start with AAindex Dnn
        aaindex_x = core.Flatten()(aaindex_input)
        attention_probs = Dense(aaindex_row*aaindex_col, activation='softmax', name='5')(aaindex_x)
        aaindex_x = Multiply()([aaindex_x, attention_probs])
        aaindex_x = BatchNormalization()(aaindex_x)
        aaindex_x = Dense(256, kernel_initializer='he_uniform', activation='relu', name='6')(aaindex_x)
        aaindex_x = Dropout(0.6)(aaindex_x)
        aaindex_x = Dense(128, kernel_initializer='he_uniform', activation='softplus', name='7')(aaindex_x)
        # aaindex_x = BatchNormalization()(aaindex_x)
        aaindex_x = Dropout(0.55)(aaindex_x)

        aaindex_x = GaussianNoise(10)(aaindex_x)

        output_aaindex_x = Dense(64, kernel_initializer='glorot_normal', activation='relu', name='8')(aaindex_x)

        # output_aaindex_x = Dense(nb_classes, kernel_initializer='glorot_normal', activation='softmax',
        #                          kernel_regularizer=l2(0.001), name='19')(output_aaindex_x)

        # output = Maximum()([output_ook_x, output_aaindex_x])
        output = Concatenate()([output_ook_x, output_aaindex_x])
        output = BatchNormalization()(output)
        
        # output = Dense(64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001), name="21")(output)
        # output = BatchNormalization()(output)
        output = Dense(128, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001), name="9")(output)
        output = Dropout(0.6)(output)
        output = Dense(64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001), name="10")(output)
        output = Dropout(0.5)(output)
        output = Dense(16, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001), name="11")(output)
        out = Dense(nb_classes, kernel_initializer='glorot_normal',
                    activation='softmax', kernel_regularizer=l2(0.001), name='12')(output)

        multinn = Model([ook_input, aaindex_input], out)
        multinn.compile(loss=keras.losses.binary_crossentropy, optimizer=optimization,
                        metrics=[keras.metrics.binary_accuracy])

    else:
        multinn = compileModel

    oneofkclass_weights = class_weight

    if (earlystop is None):
        fitHistory = multinn.fit([ook_x_t, aaindex_x_t], train_ook_Y, batch_size=batch_size, nb_epoch=nb_epoch)
    else:
        weight_checkpointer = ModelCheckpoint(filepath='temp/temp.h5', verbose=verbose, save_best_only=True,
                                              monitor='val_binary_accuracy', mode='auto', save_weights_only=True)
        fitHistory = multinn.fit([ook_x_t, aaindex_x_t], train_ook_Y, batch_size=batch_size, epochs=nb_epoch,
                                 shuffle=True,
                                 validation_split=0.2, callbacks=[early_stopping, weight_checkpointer],
                                 class_weight=oneofkclass_weights, verbose=verbose)
    return multinn


if __name__ == '__main__':
    fastaDir = "fasta/"
    fastaFileName = "sspka_general_train.fasta"
    # fastaFileName = "fortrain40.fasta"
    Train_or_Test = "Train"
    windowType = "OOK"
    windowSize = 26
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    train_ook_X, train_ook_Y = PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=0.8, random_state=0)

    windowType = "AAIndex"
    windowSize = 14
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    trainAAIndexX, train_aaindex_Y = InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=0.8, random_state=0)

    model = MultiCNN(train_ook_X, trainAAIndexX, train_ook_Y, nb_epoch=1000, earlystop=30, compiletimes=0, batch_size=2048,
                     predict=False, compileModel=None, class_weight={0: 0.55, 1: 0.45}, verbose=0)
    del train_ook_X, trainAAIndexX
    fastaFileName = "TestSet.fasta"
    Train_or_Test = "Test"
    windowType = "OOK"
    windowSize = 26
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    test_ook_x, test_ook_y = PreOneOfKey().getTestOneofkeyNLabel(positiveFile, negativeFile, random_state=0)

    windowType = "AAIndex"
    windowSize = 14
    getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
    positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
    negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
    test_aaindex_x, test_aaindex_y = InitAAindex().getTestAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)

    test_ook_x.shape = (test_ook_x.shape[0], test_ook_x.shape[2], test_ook_x.shape[3])
    test_aaindex_x.shape = (test_aaindex_x.shape[0], test_aaindex_x.shape[2], test_aaindex_x.shape[3])
    predict_probability = model.predict(x=[test_ook_x, test_aaindex_x], batch_size=1000, verbose=1)
    y_pred = predict_probability.argmax(axis=-1)
    result = Evaluator().calculate_performance(test_ook_y[:, 1], y_pred, predict_probability[:, 1])
    model_id=1
    for rs in np.random.randint(10, 10000, 10):
        fastaFileName = "sspka_general_train.fasta"
        Train_or_Test = "Train"
        windowType = "OOK"
        windowSize = 26
        getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        train_ook_X, train_ook_Y = PreOneOfKey().getTrainOneofkeyNLabel(positiveFile, negativeFile, selectedPercent=0.8,
                                                                        random_state=rs)
        windowType = "AAIndex"
        windowSize = 14
        getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        trainAAIndexX, train_aaindex_Y = InitAAindex().getTrainAAIndex(positiveFile, negativeFile, selectedPercent=0.8,
                                                                       random_state=rs)
        model = MultiCNN(train_ook_X, trainAAIndexX, train_ook_Y, nb_epoch=1000, earlystop=40, compiletimes=1,
                         batch_size=2048,
                         predict=False, compileModel=model, class_weight={0: 0.52, 1: 0.48}, verbose=0, model_id=model_id)
        model_id += 1
        del train_ook_X, trainAAIndexX
        # fastaFileName = "TestSet.fasta"
        # Train_or_Test = "Test"
        # windowType = "OOK"
        # windowSize = 26
        # getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        # positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        # negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        # test_ook_x, test_ook_y = PreOneOfKey().getTestOneofkeyNLabel(positiveFile, negativeFile, random_state=0)
        #
        # windowType = "AAIndex"
        # windowSize = 8
        # getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
        # positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
        # negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
        # test_aaindex_x, test_aaindex_y = InitAAindex().getTestAAIndex(positiveFile, negativeFile, selectedPercent=1,
        #                                                               random_state=0)
        #
        # test_ook_x.shape = (test_ook_x.shape[0], test_ook_x.shape[2], test_ook_x.shape[3])
        # test_aaindex_x.shape = (test_aaindex_x.shape[0], test_aaindex_x.shape[2], test_aaindex_x.shape[3])
        model.save("model/"+str(model_id)+".h5")
        predict_probability = model.predict(x=[test_ook_x, test_aaindex_x], batch_size=1000, verbose=1)
        y_pred = predict_probability.argmax(axis=-1)
        result = Evaluator().calculate_performance(test_ook_y[:, 1], y_pred, predict_probability[:, 1])


        # trainX, trainY = getTrainOneofkeyNLabel(fileType+"PosSequence.seq", fileType+"NegSequence.seq", selectedPercent=1)
