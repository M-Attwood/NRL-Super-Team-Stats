"""Single source of truth for filesystem paths.

Anchoring everything to PROJECT_ROOT means scripts work the same whether
launched from the project directory, a cron entry, an IDE run config, or
anywhere else — no more cwd-dependent surprises.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_ROUNDS = DATA_DIR / "rounds"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_INPUTS = DATA_DIR / "inputs"  # per-round YAML squad/starters input files

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MASTER_HISTORICAL_CSV = DATA_PROCESSED / "master_historical.csv"
