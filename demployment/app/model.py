import joblib
from pathlib import Path

__version__ = "0.1"
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(
    f"{BASE_DIR}/trained_models/final_model_accepted_rejected-{__version__}.joblib",
    "rb",
) as f:
    model_accept_reject = joblib.load(f)

with open(
    f"{BASE_DIR}/trained_models/model_single_grade-{__version__}.joblib", "rb"
) as f:
    model_single_grade = joblib.load(f)

with open(
    f"{BASE_DIR}/trained_models/model_single_sugrade-{__version__}.joblib", "rb"
) as f:
    model_single_subgrade = joblib.load(f)

with open(
    f"{BASE_DIR}/trained_models/model_single_intr-{__version__}.joblib", "rb"
) as f:
    model_single_intr = joblib.load(f)

with open(
    f"{BASE_DIR}/trained_models/model_joint_grade-{__version__}.joblib", "rb"
) as f:
    model_joint_grade = joblib.load(f)

with open(
    f"{BASE_DIR}/trained_models/model_joint_subgrade-{__version__}.joblib", "rb"
) as f:
    model_joint_subgrade = joblib.load(f)

with open(
    f"{BASE_DIR}/trained_models/model_joint_intr-{__version__}.joblib", "rb"
) as f:
    model_joint_intr = joblib.load(f)
