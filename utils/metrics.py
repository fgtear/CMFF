import numpy as np

from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics_for_regression(y_pred, y_true):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    y_true: shape of <B>
    y_pred: shape of <B>
    """

    def multiclass_acc(preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    y_pred = y_pred.cpu().detach()
    y_true = y_true.cpu().detach()

    test_preds = np.array(y_pred)
    test_truth = np.array(y_true)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # pos - neg
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    mult_a2_pos_neg = accuracy_score(binary_truth, binary_preds)
    f_score = f1_score(binary_preds, binary_truth, average="weighted")

    # if to_print:
    #     print("mae: ", mae)
    #     print("corr: ", corr)
    #     print("mult_acc: ", mult_a7)
    #     print("Classification Report (pos/neg) :")
    #     print(classification_report(binary_truth, binary_preds, digits=5))
    #     print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))

    # non-neg - neg
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0
    mult_a2_non_neg_neg = accuracy_score(binary_truth, binary_preds)
    f1_a2_non_neg_neg = f1_score(binary_preds, binary_truth, average="weighted")

    # if to_print:
    #     print("Classification Report (non-neg/neg) :")
    #     print(classification_report(binary_truth, binary_preds, digits=5))
    #     print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

    # return accuracy_score(binary_truth, binary_preds)

    return {
        "mult_a2_non_neg_neg": mult_a2_non_neg_neg,
        "accuracy": mult_a2_pos_neg,
        "mult_a5": mult_a5,
        "mult_a7": mult_a7,
        "f1": f_score,
        "f1_a2_non_neg_neg": f1_a2_non_neg_neg,
        "mae": mae,
        "corr": corr,
    }


def compute_metrics(output_cls, label_cls):
    # output_reg = output_reg.view(-1).cpu().detach().numpy()
    output_cls = output_cls.cpu().detach().numpy()
    label_cls = label_cls.cpu().detach().numpy()
    # label = label.cpu().detach().numpy()
    # mae = np.mean(np.absolute(output_reg - label[0, :])).astype(np.float64)
    # corr = np.corrcoef(output_reg, label[0, :])[0][1]

    output_7 = np.argmax(output_cls, axis=1)
    acc7 = accuracy_score(output_7, label_cls)

    output_2_has0 = [0 if v <= 2 else 1 for v in output_7]
    label_2_has0 = [0 if v <= 2 else 1 for v in label_cls]
    acc2_has0 = accuracy_score(label_2_has0, output_2_has0)
    f1_has0 = f1_score(label_2_has0, output_2_has0, average="weighted")

    output_2_non0 = []
    label_2_non0 = []
    for i, v in enumerate(label_cls):
        if v == 3:
            continue
        elif v < 3:
            label_2_non0.append(0)
            output_2_non0.append(0 if output_7[i] < 3 else 1)
        else:
            label_2_non0.append(1)
            output_2_non0.append(0 if output_7[i] < 3 else 1)
    acc2_non0 = accuracy_score(label_2_non0, output_2_non0)
    f1_non0 = f1_score(label_2_non0, output_2_non0, average="weighted")

    metrics = {
        "Has0_acc_2": round(acc2_has0, 5),
        "Has0_F1_score": round(f1_has0, 5),
        "Non0_acc_2": round(acc2_non0, 5),
        "Non0_F1_score": round(f1_non0, 5),
        "Mult_acc_7": round(acc7, 5),
        # "MAE": round(mae, 5),
        # "Corr": round(corr, 5),
    }
    return metrics


class MetricsTop:
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                "MOSI": self.__eval_mosi_regression,
                "MOSEI": self.__eval_mosei_regression,
                "SIMS": self.__eval_sims_regression,
            }
        else:
            self.metrics_dict = {
                "MOSI": self.__eval_mosi_classification,
                "MOSEI": self.__eval_mosei_classification,
                "SIMS": self.__eval_sims_classification,
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }
        """
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()
        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average="weighted")
        # two classes
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<= 0 or > 0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average="weighted")
        # without 0 (< 0 or > 0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_pred_2, average="weighted")

        eval_results = {
            "Has0_acc_2": round(Has0_acc_2, 4),
            "Has0_F1_score": round(Has0_F1_score, 4),
            "Non0_acc_2": round(Non0_acc_2, 4),
            "Non0_F1_score": round(Non0_F1_score, 4),
            "Acc_3": round(Mult_acc_3, 4),
            "F1_score_3": round(F1_score_3, 4),
        }
        return eval_results

    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
        test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
        test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
        test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)
        test_preds_a3 = np.clip(test_preds, a_min=-1.0, a_max=1.0)
        test_truth_a3 = np.clip(test_truth, a_min=-1.0, a_max=1.0)

        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)  # Average L1 distance between preds and truths

        corr = np.corrcoef(test_preds, test_truth)[0][1]
        # corr = np.corrcoef(test_preds, test_truth)[0][1]  # TODO:

        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = test_truth[non_zeros.astype(int)] > 0
        non_zeros_binary_preds = test_preds[non_zeros.astype(int)] > 0

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average="weighted")

        binary_truth = test_truth >= 0
        binary_preds = test_preds >= 0
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average="weighted")

        eval_results = {
            "Has0_acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            # "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).detach().numpy()
        test_truth = y_true.view(-1).detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1.0, a_max=1.0)
        test_truth = np.clip(test_truth, a_min=-1.0, a_max=1.0)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average="weighted")

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),  # Correlation Coefficient
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
