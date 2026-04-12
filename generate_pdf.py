"""
Convert the Honours Project Report from Markdown to a styled PDF.
Uses fpdf2 — no system dependencies required.
"""

from fpdf import FPDF  # type: ignore
import re
import os


def sanitize(text):
    """Replace Unicode chars with ASCII equivalents for Helvetica."""
    replacements = {
        '\u2014': '-',   # em dash
        '\u2013': '-',   # en dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...',  # ellipsis
        '\u2022': '-',   # bullet
        '\u00b2': '2',   # superscript 2
        '\u00b3': '3',   # superscript 3
        '\u00b0': ' deg ',  # degree
        '\u03b5': 'e',   # epsilon
        '\u03b3': 'g',   # gamma
        '\u03b1': 'a',   # alpha
        '\u03c0': 'pi',  # pi
        '\u2265': '>=',  # >=
        '\u2264': '<=',  # <=
        '\u00d7': 'x',   # multiplication
        '\u00b1': '+/-', # plus-minus
        '\u221a': 'sqrt', # sqrt
        '\u03a3': 'SUM', # Sigma
        '\u2248': '~',   # approximately
        '\u2260': '!=',  # not equal
        '\xb2': '2',
        '\xb3': '3',
        '\xb0': ' deg ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.expanduser("~/.gemini/antigravity/brain/8278f325-9377-491e-a96e-e24c18a2d705")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

# ─── Read the markdown ───
md_path = os.path.join(REPORT_DIR, "honours_project_report.md")
with open(md_path, "r") as f:
    md_lines = f.readlines()


class ReportPDF(FPDF):
    """Custom PDF with header/footer."""
    
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Honours Project - AI/ML-Based Ablation Zone Prediction", align="C")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.add_page()
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(26, 60, 94)
            self.cell(0, 14, sanitize(title), new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(26, 60, 94)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(6)
        elif level == 2:
            self.ln(6)
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(44, 82, 130)
            self.cell(0, 12, sanitize(title), new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(203, 213, 224)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)
        elif level == 3:
            self.ln(4)
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(45, 55, 72)
            self.cell(0, 10, sanitize(title), new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 4:
            self.ln(3)
            self.set_font("Helvetica", "BI", 11)
            self.set_text_color(74, 85, 104)
            self.cell(0, 9, sanitize(title), new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(26, 26, 26)
        # Handle bold and code in text
        text = text.strip()
        if not text:
            return
        self.multi_cell(0, 5.5, sanitize(text))
        self.ln(2)

    def blockquote(self, text):
        self.set_fill_color(235, 248, 255)
        self.set_draw_color(44, 82, 130)
        x = self.get_x()
        y = self.get_y()
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(45, 55, 72)
        self.set_x(x + 8)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 12, 5.5, sanitize(text), fill=True)
        # Draw left border
        self.line(x + 4, y, x + 4, self.get_y())
        self.ln(3)

    def add_image_safe(self, path):
        """Add image scaled to page width."""
        if not os.path.exists(path):
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(200, 0, 0)
            self.cell(0, 8, f"[Image not found: {os.path.basename(path)}]", new_x="LMARGIN", new_y="NEXT")
            return
        
        # Check if we need a new page
        if self.get_y() > self.h - 100:
            self.add_page()
        
        img_width = self.w - self.l_margin - self.r_margin - 10
        self.image(path, x=self.l_margin + 5, w=img_width)
        self.ln(5)

    def add_table(self, headers, rows):
        """Add a formatted table."""
        self.ln(3)
        col_count = len(headers)
        page_width = self.w - self.l_margin - self.r_margin
        col_width = page_width / col_count

        # Limit col_width for tables with few columns
        if col_count <= 3:
            col_width = min(col_width, 60)
        
        # Header
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(44, 82, 130)
        self.set_text_color(255, 255, 255)
        for h in headers:
            self.cell(col_width, 7, sanitize(h.strip()), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(247, 250, 252)
            else:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(26, 26, 26)
            for cell in row:
                self.cell(col_width, 6, sanitize(cell.strip()), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(3)

    def bullet_item(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(26, 26, 26)
        x = self.l_margin + indent * 6
        self.set_x(x)
        bullet = "-  "
        self.multi_cell(self.w - x - self.r_margin, 5.5, sanitize(bullet + text))
        self.ln(1)

    def numbered_item(self, number, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(26, 26, 26)
        self.multi_cell(0, 5.5, sanitize(f"{number}. {text}"))
        self.ln(1)


def clean_md_text(text):
    """Remove markdown formatting for plain text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
    text = re.sub(r'`(.+?)`', r'\1', text)  # code
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # links
    text = re.sub(r'[⭐🏆📊📁▶✅]', '', text)  # emojis
    return text.strip()


def parse_table(lines, start_idx):
    """Parse a markdown table starting at start_idx."""
    headers = [c.strip() for c in lines[start_idx].strip().strip('|').split('|')]
    rows = []
    i = start_idx + 2  # Skip separator line
    while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
        row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
        rows.append(row)
        i += 1
    return headers, rows, i


# ─── Build PDF ───
pdf = ReportPDF("P", "mm", "A4")
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# Title page
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(26, 60, 94)
pdf.multi_cell(0, 14, "AI/ML-Based Predictive Model\nfor Ablation Zone Estimation\nin Tumor Treatment", align="C")
pdf.ln(15)
pdf.set_font("Helvetica", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Honours Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(30)
pdf.set_draw_color(26, 60, 94)
pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "I", 11)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "March 2026", align="C")

# Process markdown
i = 0
in_blockquote = False
blockquote_text = ""
skip_until = -1

while i < len(md_lines):
    line = md_lines[i]
    stripped = line.strip()
    
    if i < skip_until:
        i += 1
        continue

    # Skip the title (already on cover)
    if stripped.startswith("# AI/ML-Based") or stripped == "**Honours Project Report**":
        i += 1
        continue
    
    # Skip TOC
    if stripped == "## Table of Contents":
        # Skip until next ## heading
        i += 1
        while i < len(md_lines) and not (md_lines[i].strip().startswith("## ") and "Table of Contents" not in md_lines[i]):
            if md_lines[i].strip().startswith("## "):
                break
            i += 1
        continue

    # Horizontal rules
    if stripped == "---":
        i += 1
        continue

    # Headers
    if stripped.startswith("#### "):
        title = clean_md_text(stripped[5:])
        pdf.chapter_title(title, level=4)
        i += 1
        continue
    elif stripped.startswith("### "):
        title = clean_md_text(stripped[4:])
        pdf.chapter_title(title, level=3)
        i += 1
        continue
    elif stripped.startswith("## "):
        title = clean_md_text(stripped[3:])
        pdf.chapter_title(title, level=1)
        i += 1
        continue

    # Images
    img_match = re.match(r'!\[.*?\]\((.+?)\)', stripped)
    if img_match:
        img_path = img_match.group(1)
        # Also check plots dir
        if not os.path.exists(img_path):
            basename = os.path.basename(img_path)
            img_path = os.path.join(PLOTS_DIR, basename)
        pdf.add_image_safe(img_path)
        i += 1
        continue

    # Tables
    if '|' in stripped and stripped.startswith('|') and i + 1 < len(md_lines) and '---' in md_lines[i + 1]:
        headers, rows, next_i = parse_table(md_lines, i)
        if headers and rows:
            pdf.add_table(headers, rows)
        i = next_i
        continue

    # Blockquotes
    if stripped.startswith('> '):
        text = clean_md_text(stripped[2:])
        # Check for multi-line blockquote
        quote_lines = [text]
        j = i + 1
        while j < len(md_lines) and md_lines[j].strip().startswith('> '):
            quote_lines.append(clean_md_text(md_lines[j].strip()[2:]))
            j += 1
        pdf.blockquote(' '.join(quote_lines))
        i = j
        continue

    # Bullet points
    if stripped.startswith('- ') or stripped.startswith('* '):
        text = clean_md_text(stripped[2:])
        indent = (len(line) - len(line.lstrip())) // 2
        pdf.bullet_item(text, indent)
        i += 1
        continue

    # Numbered list
    num_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
    if num_match:
        pdf.numbered_item(num_match.group(1), clean_md_text(num_match.group(2)))
        i += 1
        continue

    # Regular text
    if stripped and not stripped.startswith('```'):
        text = clean_md_text(stripped)
        if text:
            pdf.body_text(text)
    
    i += 1


# Save
output_path = os.path.join(SCRIPT_DIR, "Honours_Project_Report.pdf")
pdf.output(output_path)
print(f"✅ PDF saved to: {output_path}")
print(f"   File size: {os.path.getsize(output_path) / 1024:.0f} KB")
