# filtering_models/config.py
import sys
from pathlib import Path

# 1. Figure out where config.py lives
ROOT = Path(__file__).resolve().parent

# 2. Point to the directory that contains OptimizedDataGenerator4.py
#    (adjust the relative path as needed)
ODG_DIR = ROOT.parent

# 3. Stick that folder at the front of sys.path
if str(ODG_DIR) not in sys.path:
    sys.path.insert(0, str(ODG_DIR))

# 4. Your other config values:
BASE_DIR       = ROOT / "filtering_models/filtering_records2000"
TRAIN_DIR      = BASE_DIR / "tfrecords_train"
VALIDATION_DIR = BASE_DIR / "tfrecords_validation"