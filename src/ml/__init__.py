# =============================================================================
# src/ml — 机器学习资产定价模块
#
# 参考：Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning",
#       Review of Financial Studies 33(5), 2223-2273.
#
# 设计原则：**只做预测，不碰既有回测链路**。
#   本模块的产出是一张「月末 × 股票」的预测收益宽表 ŷ，
#   直接喂给现有的 backtest.engine.group_return / metrics.calc_ic /
#   backtest.report，与单因子走完全相同的评估流程。
#   这样 ML 与既有 62 个因子的结果天然可比，且不需要改动任何既有代码。
#
# 模块划分：
#   dataset.py   宽表因子 → 长面板（截面 rank 标准化 + 标签对齐）
#   cv.py        滚动窗口切分（训练/验证/测试严格时序隔离）
#   models.py    模型库（线性 → 降维 → 树），统一 fit/predict 接口
#   evaluate.py  R²_oos / Diebold-Mariano / 变量重要性 / 张成检验
# =============================================================================
