import numpy as np


def nan_mse(test, pred):
    mask = ~np.isnan(test)
    return ((test[mask] - pred[mask]) ** 2).sum() / mask.sum()


def nan_rmse(test, pred):
    return np.sqrt(nan_mse(test, pred))


def rmsle(test, pred):
    return ((np.log(test) - np.log(pred)) ** 2).sum() / test.size


def nan_rmsle(test, pred):
    mask = ~np.isnan(test)
    return rmsle(test[mask], pred[mask])


def nan_r2_score(test, pred):
    return 1 - nan_mse(test, pred) / np.nanvar(test)


def evaluate_model(regressor, sequences, length, metric=nan_mse, threshold=1.0):
    predicts = np.copy(sequences)
    for i in range(length, predicts.shape[1]):
        predicts[:, i] = np.maximum(threshold, regressor.predict(np.nan_to_num(predicts[:, i - length:i])))
    return metric(sequences[:, length:], predicts[:, length:])