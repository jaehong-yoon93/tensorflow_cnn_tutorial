import numpy as np

def accuracy(preds, labels):
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1))
          / preds.shape[0])
