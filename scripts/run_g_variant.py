"""
规则 g（剔 A 股市值最小 30%）开 / 关的对照重跑，不改 settings 默认、不覆盖既有输出。

    python scripts/run_g_variant.py main --g on  --out output/g_on
    python scripts/run_g_variant.py main --g off --out output/g_off_ctrl
    python scripts/run_g_variant.py ml   --g on  --out output/g_on/ml_B [-- src.ml.run 的参数]

- 因子值读既有 output/cache/（冻结：跳过新鲜度校验），保证 g 开 / 关只差股票池，
  不混入 Database 更新带来的因子值变化。因子值本身不依赖组建日股票池，
  唯一例外是 ps_liq_beta（用了组建日筛选后的收益，待办 #21），沿用 g 关版缓存。
- 收益与 ML 标签由 calc_monthly_returns 按 FORMATION_CONFIG 现算 → g 生效。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["main", "ml"])
ap.add_argument("--g", choices=["on", "off"], required=True)
ap.add_argument("--out", required=True)
args, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]

from src.config import settings  # noqa: E402

# 原地修改：universe.build_formation_mask 每次调用时才读这个 dict
settings.FORMATION_CONFIG["exclude_bottom_size"] = (args.g == "on")
out = (ROOT / args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
print(f"[run_g_variant] g={args.g}  out={out}  FORMATION_CONFIG={settings.FORMATION_CONFIG}")

if args.what == "main":
    import src.main as m
    m.FACTOR_OUTPUT_DIR = out                       # stats / img 改写到 out；CACHE_DIR 仍是 output/cache
    m._cache_stale_reason = lambda *a, **k: None    # 冻结缓存
    m.main()
else:
    import src.ml.run as r
    r.OUT_DIR = out
    sys.exit(r.main(rest))
