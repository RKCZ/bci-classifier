import argparse
import logging
import matplotlib.pyplot as plt

from bciclassifier.classify import classify_target, classify_audiovisual
from bciclassifier.data_manager import DataManager
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

TYPES = ('audiovisual', 'target')
LOGLEVELS = ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
METRICS = ('confusion_matrix', 'recall', 'precision')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to P300 datasets')
parser.add_argument('-l', '--loglevel', choices=LOGLEVELS, help="logging level")
parser.add_argument('-t', '--type', choices=TYPES, help="type of classification")
parser.add_argument('-m', '--metrics', choices=METRICS, help="which metrics should be evaluated")
args = parser.parse_args()

# Configure logging
if args.loglevel is None:
    logging.basicConfig()
else:
    logging.basicConfig(level=args.loglevel)
logging.debug(f"Command line arguments: {args}")

# Set metrics
metrics = args.metrics
if metrics is None:
    metrics = METRICS

dm = DataManager(args.path)

result = {}
if args.type == TYPES[0]:
    result[TYPES[0]] = classify_audiovisual(dm, metrics)
elif args.type == TYPES[1]:
    result[TYPES[1]] = classify_target(dm, metrics)
else:
    result[TYPES[1]] = classify_target(dm, metrics)
    result[TYPES[0]] = classify_audiovisual(dm, metrics)

for res in result:
    print(f"{res}: {result[res]}")
    if "confusion_matrix" in result[res]:
        ConfusionMatrixDisplay(result[res]["confusion_matrix"], display_labels=None).plot()
    if "recall" in result[res] and "precision" in result[res]:
        PrecisionRecallDisplay(precision=result[res]["precision"], recall=result[res]["recall"]).plot()
    plt.show()
