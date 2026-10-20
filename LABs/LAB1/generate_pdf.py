"""将 report.md 转换为高质量排版的 report.pdf。

流程：
1. 读取 report.md 并解析（保护数学公式中的 $ 和 $$）；
2. 使用 Python markdown 库渲染为 HTML；
3. 注入 MathJax 3 和 GitHub-style 排版 CSS，自动将 figures/*.png 转为绝对路径；
4. 调用 headless Google Chrome 将 HTML 打印为高质量 report.pdf。
"""

import os
import re
import subprocess
import base64
from pathlib import Path
import markdown

HERE = Path(__file__).parent.resolve()
MD_PATH = HERE / "report.md"
HTML_PATH = HERE / "report.html"
PDF_PATH = HERE / "report.pdf"

with open(MD_PATH, "r", encoding="utf-8") as f:
    raw_md = f.read()

# 保护行间和行内公式，防止被 markdown 解析器篡改
math_blocks = []

def save_block_math(match):
    idx = len(math_blocks)
    math_blocks.append(match.group(0))
    return f"@@BLOCKMATH{idx}@@"

def save_inline_math(match):
    idx = len(math_blocks)
    math_blocks.append(match.group(0))
    return f"@@INLINEMATH{idx}@@"

# 保护 $$ ... $$
content = re.sub(r"\$\$(.*?)\$\$", save_block_math, raw_md, flags=re.DOTALL)
# 保护 $ ... $
content = re.sub(r"\$([^\$\n]+?)\$", save_inline_math, content)

# 将图片路径相对转换为绝对路径
def fix_image_path(match):
    alt = match.group(1)
    src = match.group(2)
    img_path = (HERE / src).resolve()
    if img_path.exists():
        return f"![{alt}]({img_path.as_uri()})"
    return match.group(0)

content = re.sub(r"!\[(.*?)\]\((.*?)\)", fix_image_path, content)

# 渲染 markdown
html_body = markdown.markdown(content, extensions=["tables", "fenced_code"])

# 恢复公式
for idx, math in enumerate(math_blocks):
    html_body = html_body.replace(f"@@BLOCKMATH{idx}@@", math)
    html_body = html_body.replace(f"@@INLINEMATH{idx}@@", math)

# 完整 HTML 模板
html_full = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>FAIML Lab1 实验报告</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
<style>
@page {{
  size: A4;
  margin: 22mm 18mm 22mm 18mm;
  @bottom-right {{
    content: counter(page);
  }}
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Serif CJK SC", "Source Han Serif SC", serif;
  font-size: 14px;
  line-height: 1.68;
  color: #24292e;
  max-width: 900px;
  margin: 0 auto;
  padding: 10px;
}}
h1, h2, h3, h4 {{
  color: #111;
  font-weight: 600;
  page-break-after: avoid;
}}
h1 {{
  font-size: 24px;
  text-align: center;
  border-bottom: 2px solid #eaecef;
  padding-bottom: 12px;
  margin-top: 20px;
}}
h2 {{
  font-size: 18px;
  border-bottom: 1px solid #eaecef;
  padding-bottom: 6px;
  margin-top: 26px;
}}
h3 {{
  font-size: 15px;
  margin-top: 18px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0;
  font-size: 13px;
  page-break-inside: avoid;
}}
th, td {{
  border: 1px solid #d1d5da;
  padding: 6px 10px;
  text-align: left;
}}
th {{
  background-color: #f6f8fa;
  font-weight: 600;
}}
tr:nth-child(even) {{
  background-color: #fcfcfc;
}}
img {{
  max-width: 95%;
  height: auto;
  display: block;
  margin: 14px auto;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  page-break-inside: avoid;
}}
pre {{
  background-color: #f6f8fa;
  border-radius: 4px;
  padding: 10px 14px;
  overflow-x: auto;
  font-size: 12.5px;
  border: 1px solid #e1e4e8;
  page-break-inside: avoid;
}}
code {{
  font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace;
  background-color: rgba(27,31,35,0.05);
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 88%;
}}
pre code {{
  background-color: transparent;
  padding: 0;
}}
blockquote {{
  border-left: 4px solid #0366d6;
  margin: 12px 0;
  padding: 4px 12px;
  color: #586069;
  background-color: #f1f8ff;
}}
.MathJax {{
  font-size: 1.05em !important;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_full)
print(f"Generated HTML: {HTML_PATH}")

# 使用 Chrome headless 打印为 PDF
chrome_cmd = [
    "google-chrome",
    "--headless",
    "--no-sandbox",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=6000",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH.as_uri()
]

print("Executing Chrome headless PDF export...")
res = subprocess.run(chrome_cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("Stderr (first 300 chars):", res.stderr[:300])

if PDF_PATH.exists():
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"SUCCESS: Generated PDF at {PDF_PATH} ({size_kb:.1f} KB)")
else:
    print(f"ERROR: Failed to generate PDF at {PDF_PATH}")

