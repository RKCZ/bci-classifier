import argparse
import logging

from bciclassifier.classify import classify_target, classify_audiovisual
from bciclassifier.data_manager import DataManager

TYPES = ('audiovisual', 'target')
LOGLEVELS = ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to P300 datasets')
parser.add_argument('-l', '--loglevel', choices=LOGLEVELS, help="logging level")
parser.add_argument('-t', '--type', choices=TYPES, help="type of classification")
args = parser.parse_args()

# Configure logging
if args.loglevel is None:
    logging.basicConfig()
else:
    logging.basicConfig(level=args.loglevel)
logging.debug(f"Command line arguments: {args}")

dm = DataManager(args.path)

if args.type == TYPES[0]:
    classify_audiovisual(dm)
elif args.type == TYPES[1]:
    classify_target(dm)
else:
    classify_target(dm)
    classify_audiovisual(dm)
