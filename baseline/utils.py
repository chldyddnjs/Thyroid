import numpy as np
import torch
import torch.nn.functional as F


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu

def add_hist(hist, label_trues, label_preds, n_class,threshold=0.5):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class,threshold)

    return hist

def _fast_hist(label_true, label_pred, n_class,threshold):
    
    label_pred[label_pred >= threshold] = np.float16(1)
    label_pred[label_pred < threshold] = np.float16(0)
    #스페어
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def accuracy_check(mask, prediction, threshold=0.5):
    
    prediction[prediction >  threshold] = float(1)
    prediction[prediction < threshold] = float(0)

    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        item = item.detach().cpu().numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())

def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(0,batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc/batch_size