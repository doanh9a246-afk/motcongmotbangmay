#!/usr/bin/env python3
"""Ghep bao_cao/ch1..ch5 thanh mot file PDF (trang bia, muc luc, tai lieu tham khao).

    python scripts/build_report.py            # -> bao_cao/bao_cao_nhom.pdf
    python scripts/build_report.py --html-only

Can: Python 'markdown'; Google Chrome hoac Microsoft Edge (in PDF headless);
'pypdf' (tuy chon) de dien so trang vao muc luc. Cong thuc ve bang KaTeX (CDN),
nen may can co mang khi build.
"""
import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
BAO_CAO = ROOT / "bao_cao"
REPO_URL = "https://github.com/doanh9a246-afk/motcongmotbangmay"

CHAPTERS = [
    "ch1_gioi_thieu.md",
    "ch2_co_so_ly_thuyet_thuat_toan_tuan_tu.md",
    "ch3_openmp.md",
    "ch4_cuda.md",
    "ch5_so_sanh_ket_luan.md",
]

# ---- Trang bia: dien thong tin truong / hoc phan / giang vien vao day ----
COVER = {
    "school": "TRƯỜNG ……………………………………………",
    "faculty": "KHOA ……………………………………………",
    "course": "Học phần: ……………………………………………",
    "lecturer": "……………………………………………",
    "place_date": "Tháng 11 năm 2026",
}
MEMBERS = [
    ("Nguyễn Quốc Doanh", "Nhóm trưởng, tích hợp, benchmark tổng hợp", "Ch.1, Ch.5"),
    ("Vũ Văn Dương", "Thuật toán và bản tuần tự", "Ch.2"),
    ("Nguyễn Tùng Dương", "Song song hóa OpenMP", "Ch.3"),
    ("Trần Tiến Đạt", "Song song hóa CUDA", "Ch.4"),
]

REFERENCES = [
    'L. Nyland, M. Harris, J. Prins, “Fast N-Body Simulation with CUDA,” trong H. Nguyen (chủ biên), '
    '<i>GPU Gems 3</i>, Chương 31, Addison-Wesley, 2007. '
    'https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda',
    'S. J. Aarseth, M. Hénon, R. Wielen, “A comparison of numerical methods for the study of star cluster dynamics,” '
    '<i>Astronomy and Astrophysics</i>, tập 37, tr. 183–187, 1974.',
    'H. C. Plummer, “On the problem of distribution in globular star clusters,” '
    '<i>Monthly Notices of the Royal Astronomical Society</i>, tập 71, tr. 460–470, 1911.',
    'G. M. Amdahl, “Validity of the single processor approach to achieving large scale computing capabilities,” '
    '<i>AFIPS Spring Joint Computer Conference</i>, 1967.',
    'J. Barnes, P. Hut, “A hierarchical O(N log N) force-calculation algorithm,” <i>Nature</i>, tập 324, tr. 446–449, 1986.',
    'OpenMP Architecture Review Board, <i>OpenMP Application Programming Interface</i>, phiên bản 5.2, 2021. '
    'https://www.openmp.org/specifications/',
    'NVIDIA Corporation, <i>CUDA C++ Programming Guide</i>. https://docs.nvidia.com/cuda/cuda-c-programming-guide/',
    'NVIDIA Corporation, <i>NVIDIA T4 Tensor Core GPU Datasheet</i>. https://www.nvidia.com/en-us/data-center/tesla-t4/',
    f'Mã nguồn, số liệu thô và đồ thị của đề tài: {REPO_URL}',
]

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

# --------------------------------------------------------------------------

MATH_PATTERNS = [
    (re.compile(r"\$\$(.+?)\$\$", re.S), True),
    (re.compile(r"\\\[(.+?)\\\]", re.S), True),
    (re.compile(r"\\\((.+?)\\\)", re.S), False),
    (re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\$)"), False),
]


LIST_RE = re.compile(r"^( *)([-*+]|\d+\.)\s")


def normalize_lists(text):
    """Python-Markdown kho hon GitHub: can dong trong truoc danh sach va thut le 4 dau cach."""
    out, in_code, prev = [], False, ""
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else LIST_RE.match(line)
        if m:
            k = len(m.group(1))
            if 0 < k < 4:
                line = "    " + line[k:]
            elif 4 < k < 8:
                line = "        " + line[k:]
            if prev.strip() and not LIST_RE.match(prev) and not prev.startswith(" ") and not prev.startswith("|"):
                out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def protect_math(text, store):
    """Thay cong thuc bang placeholder (bo qua code block va inline code)."""
    out = []
    # tach code fence
    parts = re.split(r"(^```.*?^```[ \t]*$)", text, flags=re.M | re.S)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        sub = re.split(r"(`[^`\n]*`)", part)
        for j, s in enumerate(sub):
            if j % 2 == 0:
                for pat, display in MATH_PATTERNS:
                    def repl(m, display=display):
                        store.append((m.group(1).strip(), display))
                        key = f"MATHPH{len(store) - 1}X"
                        return f"\n\n{key}\n\n" if display else key
                    s = pat.sub(repl, s)
            out.append(s)
    return "".join(out)


def restore_math(h, store):
    def repl(m):
        tex, display = store[int(m.group(1))]
        tag = "div" if display else "span"
        return f'<{tag} class="math" data-display="{int(display)}">{html.escape(tex)}</{tag}>'
    h = re.sub(r"<p>MATHPH(\d+)X</p>", repl, h)
    return re.sub(r"MATHPH(\d+)X", repl, h)


def slug(s):
    return "s-" + re.sub(r"[^0-9a-z]+", "-", s.lower()).strip("-")


def chapter_html(name, idx, store, toc):
    text = (BAO_CAO / name).read_text(encoding="utf-8")
    # muc tai lieu tham khao rieng cua chuong -> gop vao danh muc chung
    text = re.split(r"^## Tài liệu tham khảo\s*$", text, flags=re.M)[0]
    text = protect_math(normalize_lists(text), store)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    h = md.convert(text)
    h = restore_math(h, store)

    # heading id + muc luc (h1, h2)
    counter = [0]

    def head(m):
        lvl, inner = int(m.group(1)), m.group(2)
        counter[0] += 1
        hid = f"c{idx}-{counter[0]}"
        if lvl <= 2:
            plain = re.sub(r"<[^>]+>", "", inner)
            toc.append((lvl, hid, html.unescape(plain)))
        return f'<h{lvl} id="{hid}">{inner}</h{lvl}>'
    h = re.sub(r"<h([1-4])>(.*?)</h\1>", head, h, flags=re.S)

    # chu thich hinh/bang
    h = re.sub(r"<p><em>((?:Hình|Bảng) [0-9].*?)</em></p>", r'<p class="caption">\1</p>', h, flags=re.S)
    h = re.sub(r"<p><em>(Người viết:.*?)</em></p>", r'<p class="author">\1</p>', h)
    # anh: ../results/x -> duong dan tuyet doi
    h = re.sub(r'<p>(<img [^>]*>)</p>', r'<figure>\1</figure>', h)
    h = re.sub(r'src="\.\./([^"]+)"', lambda m: f'src="{(ROOT / m.group(1)).as_uri()}"', h)
    # link tuong doi trong repo -> GitHub
    h = re.sub(r'href="\.\./([^"#]+)"', lambda m: f'href="{REPO_URL}/blob/main/{m.group(1)}"', h)
    return f'<section class="chapter">{h}</section>'


def build_html(page_of=None):
    store, toc = [], []
    chapters = [chapter_html(n, i, store, toc) for i, n in enumerate(CHAPTERS, 1)]

    rows = "".join(
        f"<tr><td>{i}</td><td>{n}</td><td>{r}</td><td>{c}</td></tr>" for i, (n, r, c) in enumerate(MEMBERS, 1)
    )
    cover = f"""
<section class="cover">
  <div class="c-top">{COVER['school']}<br>{COVER['faculty']}</div>
  <div class="c-line"></div>
  <div class="c-kind">BÁO CÁO BÀI TẬP LỚN</div>
  <div class="c-course">{COVER['course']}</div>
  <div class="c-label">Đề tài</div>
  <div class="c-title">SONG SONG HÓA MÔ PHỎNG N-BODY<br>(LỰC HẤP DẪN) BẰNG OPENMP VÀ CUDA</div>
  <div class="c-lect">Giảng viên hướng dẫn: {COVER['lecturer']}</div>
  <div class="c-label">Nhóm sinh viên thực hiện</div>
  <table class="c-members"><thead><tr><th>STT</th><th>Họ và tên</th><th>Nhiệm vụ</th><th>Phụ trách</th></tr></thead>
  <tbody>{rows}</tbody></table>
  <div class="c-date">{COVER['place_date']}</div>
</section>"""

    toc_items = toc + [(1, "refs", "Tài liệu tham khảo")]
    lis = []
    for lvl, hid, title in toc_items:
        pg = page_of.get(hid, "") if page_of else ""
        lis.append(
            f'<li class="l{lvl}"><a href="#{hid}"><span class="t">{html.escape(title)}</span>'
            f'<span class="dots"></span><span class="p">{pg}</span></a></li>'
        )
    toc_html = f'<section class="toc"><h1 class="toc-h">MỤC LỤC</h1><ul>{"".join(lis)}</ul></section>'

    refs = "".join(f"<li>{r}</li>" for r in REFERENCES)
    refs_html = f'<section class="chapter"><h1 id="refs">Tài liệu tham khảo</h1><ol class="refs">{refs}</ol></section>'

    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>Báo cáo — Song song hóa mô phỏng N-body bằng OpenMP và CUDA</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<style>{CSS}</style></head>
<body>
{cover}
{toc_html}
{''.join(chapters)}
{refs_html}
<script>
document.querySelectorAll('.math').forEach(function (el) {{
  try {{ katex.render(el.textContent, el, {{displayMode: el.dataset.display === '1', throwOnError: false}}); }}
  catch (e) {{ el.classList.add('math-err'); }}
}});
document.title = document.title;  // xong
</script>
</body></html>"""


CSS = r"""
@page { size: A4; margin: 2cm 2cm 2.2cm 3cm;
  @bottom-center { content: counter(page); font: 11pt "Times New Roman", serif; } }
@page :first { @bottom-center { content: none; } }
@page toc { @bottom-center { content: none; } }
html { font-family: "Times New Roman", Times, serif; font-size: 13pt; line-height: 1.5; color: #000; }
body { margin: 0; }
p, li { text-align: justify; hyphens: auto; }
p { margin: 0 0 0.5em; }
h1 { font-size: 16pt; text-align: center; text-transform: uppercase; margin: 0 0 0.3em; break-before: page; }
h2 { font-size: 14pt; margin: 1em 0 0.4em; break-after: avoid; }
h3 { font-size: 13pt; margin: 0.9em 0 0.3em; break-after: avoid; }
h4 { font-size: 13pt; font-style: italic; margin: 0.8em 0 0.3em; break-after: avoid; }
.author { text-align: center; font-style: italic; margin-bottom: 1.2em; }
code { font-family: Consolas, "Courier New", monospace; font-size: 10.5pt; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 6px 9px; font-size: 10pt; line-height: 1.3;
  white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { font-size: inherit; }
table { border-collapse: collapse; margin: 0.6em auto; font-size: 11pt; line-height: 1.3; break-inside: avoid; }
th, td { border: 1px solid #555; padding: 3px 7px; }
th { background: #eee; }
td { text-align: inherit; }
figure { margin: 0.8em 0 0.2em; text-align: center; break-inside: avoid; break-after: avoid; }
figure img { max-width: 100%; max-height: 11cm; }
.caption { text-align: center; font-style: italic; font-size: 12pt; margin-bottom: 1em; }
.math-err { color: #b00; }
.katex { font-size: 1.05em; }
.katex-display { margin: 0.5em 0; overflow: hidden; }
a { color: inherit; text-decoration: none; }
blockquote { margin: 0.5em 1.5em; font-style: italic; }

.cover { page: cover; height: 25.3cm; box-sizing: border-box; border: 3px double #000; padding: 1.2cm 1cm;
  display: flex; flex-direction: column; align-items: center; text-align: center; }
.c-top { font-weight: bold; font-size: 14pt; }
.c-line { width: 40%; border-top: 1px solid #000; margin: 0.4em 0 2.2cm; }
.c-kind { font-size: 20pt; font-weight: bold; }
.c-course { margin: 0.3em 0 1.6cm; }
.c-label { font-style: italic; margin-top: 0.4cm; }
.c-title { font-size: 19pt; font-weight: bold; line-height: 1.35; margin: 0.2em 0 1.4cm; }
.c-lect { margin-bottom: 0.6cm; }
.c-members { font-size: 12pt; margin-top: 0.3em; }
.c-members td { text-align: left; }
.c-date { margin-top: auto; font-weight: bold; }

.toc { page: toc; }
.toc h1 { break-before: page; }
.toc ul { list-style: none; padding: 0; margin: 0.6em 0 0; }
.toc li { text-align: left; font-size: 12pt; line-height: 1.35; }
.toc a { display: flex; align-items: baseline; }
.toc .dots { flex: 1; border-bottom: 1px dotted #000; margin: 0 4px; transform: translateY(-4px); }
.toc .l1 { font-weight: bold; margin-top: 0.5em; }
.toc .l2 { padding-left: 1.5em; }
"""


def find_browser():
    cands = [
        os.environ.get("CHROME"),
        shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in cands:
        if c and Path(c).exists():
            return c
    sys.exit("Khong tim thay Chrome/Edge. Dat bien CHROME=<duong dan>.")


def print_pdf(browser, html_path, pdf_path):
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([
            browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--user-data-dir={prof}", "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=30000", f"--print-to-pdf={pdf_path}", html_path.as_uri(),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)


def heading_pages(pdf_path):
    """Tra ve {heading id: so trang} bang cach doc text PDF (so trang = so in o chan trang)."""
    try:
        import pypdf
    except ImportError:
        return None
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [re.sub(r"\s+", " ", p.extract_text() or "") for p in reader.pages]
    _, toc = [], []
    store = []
    for i, n in enumerate(CHAPTERS, 1):
        chapter_html(n, i, store, toc)
    toc.append((1, "refs", "Tài liệu tham khảo"))
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    # bo qua trang bia + muc luc: bat dau tu trang dau tien co "Người viết:"
    start = next((i for i, p in enumerate(pages) if "người viết:" in p.lower()), 2)
    result = {}
    for _, hid, title in toc:
        key = norm(title)[:40]
        for pi in range(start, len(pages)):
            if key in pages[pi].lower():
                result[hid] = pi + 1
                start = pi
                break
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(BAO_CAO / "bao_cao_nhom.pdf"))
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    html_path = out.with_suffix(".html")

    html_path.write_text(build_html(), encoding="utf-8")
    if a.html_only:
        print(html_path)
        return
    browser = find_browser()
    print_pdf(browser, html_path, out)
    pages = heading_pages(out)
    if pages:
        html_path.write_text(build_html(pages), encoding="utf-8")
        print_pdf(browser, html_path, out)
        missing = [k for k in pages if not pages[k]]
        if missing:
            print("Canh bao: khong tim thay trang cho", missing)
    else:
        print("Khong co pypdf: muc luc khong co so trang (pip install pypdf).")
    html_path.unlink()
    print(f"Da tao {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
