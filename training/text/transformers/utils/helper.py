
# basics
import numpy as np
from sklearn.metrics import f1_score

# specific
import torch

# set classes
class_names = [str(number) for number in np.arange(1, 18)]
MAX_LEN = 512


# helper functions to load the data and model onto GPU
def get_default_device():
    """Pick GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_f1_with_optimal_thresholds(outputs, targets):
    """Find optimal thresholds and calculate best total F1-score."""

    # initialize shapes of thresholds and f1s
    thresholds = np.zeros((outputs.shape[1]))
    f1_labels = np.zeros((outputs.shape[1]))

    # transfrom from 0/1 to True/False
    targets = targets > 0.5

    # for each label get optimal thresholds
    for label in range(outputs.shape[1]):

        # select targets and ouputs of label
        tagets_label = targets[:, label].flatten()
        outputs_label = outputs[:, label].flatten()

        # caluclate best thresholds and f1
        f1_labels[label], thresholds[label] = get_f1_with_single_threshold(outputs_label, tagets_label)

    # apply optimized thresholds on outputs to get best predictions
    predictions = outputs > thresholds

    # calculate F1 with best predictions
    f1 = f1_score(predictions, targets, average='micro')

    # return F1 and thresholds
    return f1, np.around(thresholds, decimals=1).tolist()


def get_f1_with_single_threshold(outputs, targets, average='binary'):
    """Get best threshold and F1-score per label."""

    # initialize dict
    threshold_dict = {}

    # for each threshold calculate f1 score
    for t in np.arange(0, 1, 0.1):

        # apply threshold to retrieve predictions
        preds = outputs > t

        # calculate F1 and add to dict
        threshold_dict[t] = f1_score(preds, targets)

    # get mas F1 from dict and save thresholds
    best_threshold = max(threshold_dict, key=threshold_dict.get)
    best_f1 = threshold_dict[best_threshold]

    # return best F1 with best thresholds
    return best_f1, best_threshold


def get_current_ratio(df):
    """Calcualte Balancing Ratio.

    ratio = # positives in rarest label / # positives in most frequent label
    """

    # get amount of  positive in rarest label
    sdg_max = df[class_names].sum(axis=0).idxmax()
    sdg_max_n = len(df[df[sdg_max] == 1.0])

    # get amount of  positive in most frequent label
    sdg_min = df[class_names].sum(axis=0).idxmin()
    sdg_max_n = len(df[df[sdg_min] == 1.0])

    # calculate ratio
    return sdg_min / sdg_max_n
