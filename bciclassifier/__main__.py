import argparse
import logging

from bciclassifier.classify import classify_target, classify_audiovisual
from bciclassifier.data_manager import DataManager

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

if args.type == TYPES[0]:
    result = classify_audiovisual(dm, metrics)
    print(result)
elif args.type == TYPES[1]:
    result = classify_target(dm, metrics)
    print(result)
else:
    result = classify_target(dm, metrics)
    print(result)
    result = classify_audiovisual(dm, metrics)
    print(result)
