"""Quant 量化因子回测框架——源码包。入口：python -m src.main

⚠️ 下面这两行必须留在最前面、且早于任何 numpy / sklearn 的导入。
OMP_NUM_THREADS 一类的环境变量只在 BLAS / OpenMP 动态库**首次加载前**读取，
放晚了就是一句废话。把它挂在包的 __init__ 里，是唯一能保证「先于一切」的位置。
线程预算与调节方式见 src/config/compute.py。
"""

from src.config.compute import apply_thread_limits

apply_thread_limits()
