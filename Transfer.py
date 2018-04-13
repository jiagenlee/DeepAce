from keras.models import load_model
from tools import *
from PreOneofkey import PreOneOfKey
from InitAAindex import InitAAindex
from Evaluator import Evaluator
from keras.callbacks import EarlyStopping

model = load_model("model/9_save_general.h5")
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
windowType = "AAIndex"
windowSize = 14
getWindowSizePosAndNegSequence(fastaFileName, Train_or_Test, windowType, windowSize, fastaDir)
positiveFile = "seq/" + Train_or_Test + windowType + "PosSequence.seq"
negativeFile = "seq/" + Train_or_Test + windowType + "NegSequence.seq"
test_aaindex_x, test_aaindex_y = InitAAindex().getTestAAIndex(positiveFile, negativeFile, selectedPercent=1, random_state=0)
print(test_aaindex_x.shape)

test_ook_x.shape = (test_ook_x.shape[0], test_ook_x.shape[2], test_ook_x.shape[3])
test_aaindex_x.shape = (test_aaindex_x.shape[0], test_aaindex_x.shape[2], test_aaindex_x.shape[3])
predict_probability = model.predict(x=[test_ook_x, test_aaindex_x], batch_size=1000, verbose=0)

y_pred = predict_probability.argmax(axis=-1)
result = Evaluator().calculate_performance(test_ook_y[:, 1], y_pred, predict_probability[:, 1])

class_weight = {0:0.50, 1:0.50}
print(class_weight)
mcc = -1.0

for rs in np.random.randint(10, 10000, 10):

    fastaDir = "fasta/"
    fastaFileName = "fortrain40.fasta"
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
    train_ook_X.shape = (train_ook_X.shape[0], train_ook_X.shape[2], train_ook_X.shape[3])
    trainAAIndexX.shape = (trainAAIndexX.shape[0], trainAAIndexX.shape[2], trainAAIndexX.shape[3])

    model.fit([train_ook_X, trainAAIndexX], train_ook_Y, batch_size=1024, epochs=1000,
                                     shuffle=True, validation_split=0.2, callbacks=[earlystop,],
                                     class_weight=class_weight, verbose=0)

    predict_probability = model.predict(x=[test_ook_x, test_aaindex_x], batch_size=1000, verbose=0)
    y_pred = predict_probability.argmax(axis=-1)
    result = Evaluator().calculate_performance(test_ook_y[:, 1], y_pred, predict_probability[:, 1])
    if result[9] > mcc:
        mcc = result[9]
        model.save("model/mcc_" + str(mcc) + ".h5")
