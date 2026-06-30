#!/usr/bin/env python3
"""
将 A股量化因子回测研究报告.html 转换为 PDF

使用方式（在共享 venv 中运行）：
    source /Users/louis/MyProjects/venv/bin/activate
    pip install weasyprint
    python -m src.convert_report_to_pdf   （在 Quant/ 根目录执行）

输出：output/defense/A股量化因子回测研究报告.pdf
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
HTML_INPUT  = SCRIPT_DIR / "A股量化因子回测研究报告_fixed.html"
PDF_OUTPUT  = SCRIPT_DIR / "output" / "defense" / "A股量化因子回测研究报告.pdf"

PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def _try_weasyprint():
    try:
        import weasyprint
    except ImportError:
        print("weasyprint 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "weasyprint"])
        import weasyprint

    print(f"使用 weasyprint 转换: {HTML_INPUT.name} → {PDF_OUTPUT.name}")
    weasyprint.HTML(filename=str(HTML_INPUT), base_url=str(SCRIPT_DIR)).write_pdf(str(PDF_OUTPUT))
    print(f"✅ PDF 已保存至: {PDF_OUTPUT}")
    return True


def _try_chrome():
    """尝试使用本机 Chrome 的 headless 模式"""
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for chrome in chrome_paths:
        p = Path(chrome)
        if p.exists():
            print(f"使用 Chrome headless: {p.name}")
            subprocess.check_call([
                str(p),
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--print-to-pdf={PDF_OUTPUT}",
                "--print-to-pdf-no-header",
                str(HTML_INPUT),
            ])
            print(f"✅ PDF 已保存至: {PDF_OUTPUT}")
            return True
    return False


if __name__ == "__main__":
    if not HTML_INPUT.exists():
        print(f"❌ 找不到输入文件: {HTML_INPUT}")
        sys.exit(1)

    # 优先尝试 Chrome headless（无需额外依赖）
    print("尝试 Chrome headless 模式...")
    if _try_chrome():
        sys.exit(0)

    # 其次尝试 weasyprint（需要先 brew install pango）
    print("Chrome 未找到，尝试 weasyprint...")
    print("注意：weasyprint 需要系统库，如失败请先运行：brew install pango")
    if _try_weasyprint():
        sys.exit(0)

    print("\n❌ 所有自动方案均失败。")
    print("手动方法：用 Chrome/Safari 打开以下文件，选择 '打印 → 存储为 PDF'：")
    print(f"  {HTML_INPUT}")
    sys.exit(1)
