"""Single source of truth for filesystem paths.

Anchoring everything to PROJECT_ROOT means scripts work the same whether
launched from the project directory, CI, an IDE run config, or anywhere else.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"          # one scraped CSV per season + fixtures
DATA_INPUTS = DATA_DIR / "inputs"    # weekly squad / starters YAML, injuries

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
