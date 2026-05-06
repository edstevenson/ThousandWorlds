from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
__path__.append(str(ROOT / "thousandworlds"))
from .thousandworlds import *  # noqa: F401,F403

for _name in ("data", "preprocessing", "schema", "spectral"):
    sys.modules[f"{__name__}.{_name}"] = globals()[_name]

import models as _models  # noqa: E402

models = _models
sys.modules[__name__ + ".models"] = _models
