import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


class Evaluator(object):

    def calculate_performance(self, labels, predict_y, predict_score):
        result=[]
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        test_num = len(labels)
        for index in range(test_num):
            if (labels[index] == 1):
                if (labels[index] == predict_y[index]):
                    tp += 1
                else:
                    fn += 1
            else:
                if (labels[index] == predict_y[index]):
                    tn += 1
                else:
                    fp += 1
        acc = float(tp + tn) / test_num
        precision = float(tp) / (tp + fp + sys.float_info.epsilon)
        sensitivity = float(tp) / (tp + fn + sys.float_info.epsilon)
        specificity = float(tn) / (tn + fp + sys.float_info.epsilon)
        f1 = 2 * precision * sensitivity / (precision + sensitivity+sys.float_info.epsilon)
        mcc = float(tp * tn - fp * fn) / (np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
        aps = average_precision_score(labels, predict_score)
        fpr, tpr, _ = roc_curve(labels, predict_score)
        # print(fpr)
        # print(tpr)
        # np.savetxt("temp/deep_fpr_100.txt", fpr)
        # np.savetxt("temp/deep_tpr_100.txt", tpr)
        aucResults = auc(fpr, tpr)
        result.append(tp)
        result.append(fn)
        result.append(tn)
        result.append(fp)
        result.append(acc)
        result.append(precision)
        result.append(sensitivity)
        result.append(specificity)
        result.append(f1)
        result.append(mcc)
        result.append(aps)
        result.append(aucResults)
        strResults = 'tp ' + str(tp) + ' fn ' + str(fn) + ' tn ' + str(tn) + ' fp ' + str(fp)
        strResults = strResults + ' acc ' + str(acc) + ' precision ' + str(precision) + ' sensitivity ' + str(
            sensitivity)
        strResults = strResults + ' specificity ' + str(specificity) + ' f1 ' + str(f1) + ' mcc ' + str(mcc)
        strResults = strResults + ' aps ' + str(aps) + ' auc ' + str(aucResults)
        print(strResults)
        return result

    def accuracy(self, y_true, y_pred):
        pass