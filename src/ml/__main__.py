# =============================================================================
# ml/__main__.py
# 让 `python -m src.ml` 等价于 `python -m src.ml.run`
#
# 纯转发，不含任何逻辑——真正的入口与 ML_CONFIG 都在 run.py。
# =============================================================================

import sys

from src.ml.run import main

if __name__ == "__main__":
    sys.exit(main())
