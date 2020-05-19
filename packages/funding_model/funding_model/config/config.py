import funding_model
import pathlib

# Directory Paths
PACKAGE_ROOT = pathlib.Path(funding_model.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODELS_DIR = PACKAGE_ROOT / 'trained_models'
PIPELINE_FILENAME = 'lsvc_pipeline_v'

# Variable types
NUM_VARS = ['goal', 'backers_count']
DROP_VARS = ['project_id', 'name', 'deadline',
             'state_changed_at', 'created_at', 'launched_at']
CAT_VARS = ['country', 'currency']
STR_VARS = ['desc', 'keywords']
BOOL_VARS = ['disable_communication']
TARGET = 'final_status'
