#!/bin/zsh
# 会计块对照实验：A = 59 个量价 / 估值特征 vs B = 全部缓存特征（现 110 个，+48 会计块等），完整网格
#
# 唯一变量是特征集，其余（样本划分、模型、随机种子、评估口径）完全一致。
# 关键看 EW-Sign —— 它权重恒等权、不学习，它的涨幅 = 纯粹的「特征更宽」效应；
# 与 LGBM / OLS-H 涨幅之差 = 学习的净贡献。
# 样本划分、特征历史门槛等见 src/ml/run.py 的 ML_CONFIG（现行：LWZ 2022 的 9 / 3 年，样本外 2019 起）。
#
#   ./run_ml_compare.sh                 # g 关（主框架默认），9 线程
#   ./run_ml_compare.sh on              # g 开（剔 A 股市值最小 30%）
#   ./run_ml_compare.sh off 6 15        # 6 线程、窗口间散热 15 秒（机器发烫时用）
#
# 长任务请接电源或用 caffeinate -i ./run_ml_compare.sh（机器睡眠会让任务停摆，见 CLAUDE.md 待办 #28）

set -e
cd /Users/louis/MyProjects/Quant
PY=/Users/louis/MyProjects/venv/bin/python
G=${1:-off}
JOBS=${2:-9}
COOL=${3:-0}
LOG=output/ml/compare_$(date +%m%d_%H%M)_g$G
mkdir -p $LOG

BASE=$(cat output/ml/base59_features.txt)
# ⚠️ 必须用数组：zsh 不对未加引号的标量做词分割（bash 会），
#    写成 ARGS="..." 再 $ARGS 会被当成单个参数整体传进去，argparse 全部报错。
ARGS=(--full --models EW-Sign,OLS-H,LGBM --jobs $JOBS --cooldown $COOL)

echo "[$(date +%H:%M)] A：59 特征（量价 / 估值块），g $G"
$PY scripts/run_g_variant.py ml --g $G --out $LOG/A59 -- $ARGS --features "$BASE" > $LOG/A59.log 2>&1

echo "[$(date +%H:%M)] B：全部缓存特征（+ 会计块），g $G"
$PY scripts/run_g_variant.py ml --g $G --out $LOG/B -- $ARGS > $LOG/B.log 2>&1

echo "[$(date +%H:%M)] 完成 → $LOG"
paste <(echo "── A 59"; cat $LOG/A59/model_summary.csv) <(echo "── B"; cat $LOG/B/model_summary.csv)
