#!/usr/bin/env python3
"""
buat_aruco.py — pembuat marker ArUco berukuran BEBAS
=====================================================
Marker yang dihasilkan memakai kamus DICT_4X4_1000, SAMA dengan
cctv-monitor-aruco.py (ID 0-999), jadi langsung terbaca oleh program monitor.

Bentuk marker (sama dengan --make-marker di program monitor):
    6x6 kotak (bingkai hitam 1 kotak + isi 4x4)  +  tepi putih 1 kotak  =  8x8 kotak
    -> sisi hitam = 6 kotak, lebar total (dengan tepi putih) = 8 kotak.

UKURAN diberikan dengan SALAH SATU dari tiga cara (pilih yang paling mudah):
    --kotak-mm  lebar SATU kotak, dalam mm
    --sisi-cm   sisi bagian HITAM marker, dalam cm
    --total-cm  lebar total termasuk tepi putih, dalam cm

FORMAT KELUARAN (--format):
    pdf  (default)  VEKTOR, ukuran fisik pasti, tajam di ukuran apa pun (sampai ~5 m).
                    1 marker = 1 halaman. Cocok untuk printer & percetakan.
    svg             VEKTOR, tanpa batas ukuran. 1 file per marker.
    png             Raster (pakai --dpi). Hati-hati: ukuran besar = file sangat besar.

CONTOH:
    python buat_aruco.py --ids 517 --sisi-cm 12
    python buat_aruco.py --ids 517 518 519 --total-cm 30 --out marker.pdf
    python buat_aruco.py --ids 517 --kotak-mm 50 --kertas A3
    python buat_aruco.py --ids 517 --total-cm 120 --format svg          # untuk banner/percetakan
    python buat_aruco.py --ids 990 991 992 --kotak-mm 10 15 20 --out uji.pdf   # ukuran beda per ID
    python buat_aruco.py --ids 517 --sisi-cm 60 --tanpa-label           # hanya marker (percetakan)

CETAK: pilih "Actual size / 100%" (JANGAN "Fit to page"). Bila label dicetak, periksa
garis skala di bawah marker dengan penggaris. Pakai bahan doff/tidak mengkilap.

Kebutuhan: pip install opencv-python numpy     (Pillow opsional, hanya untuk metadata DPI PNG)
"""
import argparse
import os
import sys

import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_1000"     # HARUS sama dengan cctv-monitor-aruco.py
CELLS_BLACK = 6                       # 4x4 isi + bingkai hitam 1 kotak di tiap sisi
CELLS_TOTAL = 8                       # + tepi putih 1 kotak di tiap sisi
PAPERS = {"A4": (210, 297), "A3": (297, 420), "A2": (420, 594), "A1": (594, 841), "A0": (841, 1189)}
MAX_PDF_MM = 5000.0                   # batas ukuran halaman PDF (14400 pt = ~5,08 m)
MAX_PNG_PIXELS = 300_000_000
PT_PER_MM = 72.0 / 25.4


# ---------------------------------------------------------------------------
# Matriks marker (1 = hitam) dari kamus OpenCV
# ---------------------------------------------------------------------------
def marker_matrix(marker_id):
    code = getattr(cv2.aruco, ARUCO_DICT_NAME)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dic = cv2.aruco.getPredefinedDictionary(code)
    else:                                                # OpenCV lama (<4.7)
        dic = cv2.aruco.Dictionary_get(code)
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(dic, marker_id, CELLS_BLACK)
    else:
        img = cv2.aruco.drawMarker(dic, marker_id, CELLS_BLACK)
    return img < 128                                     # 6x6 bool, True = kotak hitam


# Lebar huruf Helvetica / Helvetica-Bold (ASCII 32..126, per 1000 em) -> rata tengah teks di PDF akurat
_HELV = [278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,
         556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,1015,
         667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611,
         278,278,278,469,556,333,
         556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,
         334,260,334,584]
_HELV_B = [278,333,474,556,556,889,722,238,333,333,389,584,278,333,278,278,
           556,556,556,556,556,556,556,556,556,556,333,333,584,584,584,611,975,
           722,722,722,722,667,611,778,722,278,556,722,611,833,722,778,667,778,722,667,611,722,667,944,667,667,611,
           333,278,333,584,556,333,
           556,611,556,611,556,333,611,611,278,278,556,278,889,611,611,611,611,389,556,333,611,556,778,556,556,500,
           389,280,389,584]


def text_w(s, fs, bold=False):
    """Lebar teks (mm) untuk ukuran huruf fs (mm)."""
    t = _HELV_B if bold else _HELV
    return sum(t[ord(ch) - 32] if 32 <= ord(ch) <= 126 else 556 for ch in s) / 1000.0 * fs


# ---------------------------------------------------------------------------
# Tata letak (semua dalam mm, titik asal kiri-atas) — dipakai bersama oleh PDF/SVG/PNG
# ---------------------------------------------------------------------------
def layout(marker_id, cell_mm, kertas="auto", label=True, margin_mm=None):
    total = CELLS_TOTAL * cell_mm
    black = CELLS_BLACK * cell_mm
    base = min(25.0, max(3.2, total * 0.05))             # ukuran huruf dasar (mm) ikut ukuran marker
    title = f"ID {marker_id}"
    info = [f"Sisi hitam {black / 10:.2f} cm | total dgn tepi putih {total / 10:.2f} cm",
            f"Lebar 1 kotak {cell_mm:.2f} mm | {ARUCO_DICT_NAME}"]
    title_fs, gap = base * 1.8, base * 0.5
    label_h = (gap + title_fs * 0.9 + base * 1.45 * len(info) + base * 0.3) if label else 0.0
    bar_h = base * 3.0 if label else 0.0
    content_h = total + label_h + bar_h
    m = margin_mm if margin_mm is not None else max(8.0, min(total * 0.06, 40.0))

    if kertas == "auto":
        if label:
            page_w, page_h = max(total + 2 * m, 70.0), max(content_h + 2 * m, 70.0)
        else:
            page_w, page_h = total + 2 * m, total + 2 * m
    else:
        page_w, page_h = PAPERS[kertas]
        if total + 16 > page_w or content_h + 16 > page_h:
            raise ValueError(
                f"Marker selebar {total / 10:.1f} cm tidak muat di {kertas} "
                f"({page_w / 10:.1f} x {page_h / 10:.1f} cm). Pakai --kertas auto atau kertas lebih besar.")

    x0 = (page_w - total) / 2.0
    y0 = (page_h - content_h) / 2.0
    cells = [(c, r) for r in range(CELLS_BLACK) for c in range(CELLS_BLACK) if marker_matrix(marker_id)[r, c]]

    texts, bars = [], []                                  # texts: (x_tengah, y_baseline, ukuran, tebal, isi)
    avail = page_w * 0.92

    def fit(s, fs, bold):
        w = text_w(s, fs, bold)
        return fs if w <= avail else fs * avail / w

    if label:
        ty = y0 + total + gap + title_fs * 0.9
        texts.append((page_w / 2, ty, fit(title, title_fs, True), True, title))
        for i, s in enumerate(info):
            texts.append((page_w / 2, ty + base * 1.45 * (i + 1), fit(s, base, False), False, s))
        L = 100.0 if page_w >= 116 else 50.0 if page_w >= 66 else 20.0
        line_y = y0 + total + label_h + base * 1.2
        bx = (page_w - L) / 2.0
        bars.append((bx, line_y - 0.2, L, 0.4))
        for i in range(11):
            th = 3.0 if i % 5 == 0 else 1.5
            bars.append((bx + i * L / 10 - 0.15, line_y - th / 2, 0.3, th))
        s = f"Skala: garis = {L / 10:g} cm. Cetak 100 persen (Actual size)."
        texts.append((page_w / 2, line_y + 1.5 + base * 1.1, fit(s, base, False), False, s))

    return dict(page_w=page_w, page_h=page_h, x0=x0, y0=y0, cell=cell_mm, total=total,
                cells=cells, texts=texts, bars=bars, label=label)


# ---------------------------------------------------------------------------
# PDF vektor (ditulis langsung, tanpa library tambahan)
# ---------------------------------------------------------------------------
def _esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render_pdf(pages, path):
    n = len(pages)
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offs = {}

    def emit(num, body):
        offs[num] = len(out)
        out.extend(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")

    kids = " ".join(f"{6 + 2 * i} 0 R" for i in range(n))
    emit(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    emit(2, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    emit(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    emit(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

    for i, L in enumerate(pages):
        W, H = L["page_w"] * PT_PER_MM, L["page_h"] * PT_PER_MM
        ops = ["q", "0 g"]
        c = L["cell"]
        for (cx, cy) in L["cells"]:                       # semua kotak hitam dalam 1 path -> tanpa celah antar kotak
            x = (L["x0"] + (cx + 1) * c) * PT_PER_MM
            y = H - (L["y0"] + (cy + 2) * c) * PT_PER_MM
            ops.append(f"{x:.4f} {y:.4f} {c * PT_PER_MM:.4f} {c * PT_PER_MM:.4f} re")
        for (bx, by, bw, bh) in L["bars"]:
            ops.append(f"{bx * PT_PER_MM:.4f} {H - (by + bh) * PT_PER_MM:.4f} "
                       f"{bw * PT_PER_MM:.4f} {bh * PT_PER_MM:.4f} re")
        ops.append("f")
        for (tx, ty, fs, bold, s) in L["texts"]:
            w_est = text_w(s, fs, bold)
            ops += ["BT", f"/{'F2' if bold else 'F1'} {fs * PT_PER_MM:.3f} Tf",
                    f"{(tx - w_est / 2) * PT_PER_MM:.3f} {H - ty * PT_PER_MM:.3f} Td",
                    f"({_esc(s)}) Tj", "ET"]
        ops.append("Q")
        data = "\n".join(ops).encode("latin-1", "replace")
        emit(5 + 2 * i, f"<< /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")
        emit(6 + 2 * i, (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.3f} {H:.3f}] "
                         f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                         f"/Contents {5 + 2 * i} 0 R >>").encode())

    total_objs = 5 + 2 * n
    xref = len(out)
    out.extend(f"xref\n0 {total_objs}\n".encode() + b"0000000000 65535 f \n")
    for k in range(1, total_objs):
        out.extend(f"{offs[k]:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    with open(path, "wb") as f:
        f.write(out)


# ---------------------------------------------------------------------------
# SVG vektor
# ---------------------------------------------------------------------------
def render_svg(L, path):
    W, H, c = L["page_w"], L["page_h"], L["cell"]
    d = "".join(f"M{L['x0'] + (cx + 1) * c:.4f} {L['y0'] + (cy + 1) * c:.4f}h{c:.4f}v{c:.4f}h-{c:.4f}z"
                for (cx, cy) in L["cells"])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" height="{H:.3f}mm" '
             f'viewBox="0 0 {W:.3f} {H:.3f}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<path d="{d}" fill="#000000" shape-rendering="crispEdges"/>']
    for (bx, by, bw, bh) in L["bars"]:
        parts.append(f'<rect x="{bx:.3f}" y="{by:.3f}" width="{bw:.3f}" height="{bh:.3f}" fill="#000000"/>')
    for (tx, ty, fs, bold, s) in L["texts"]:
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        parts.append(f'<text x="{tx:.3f}" y="{ty:.3f}" font-size="{fs:.3f}" text-anchor="middle" '
                     f'font-family="Helvetica, Arial, sans-serif" font-weight="{"bold" if bold else "normal"}" '
                     f'fill="#000000">{s}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# PNG raster (piksel per kotak dibulatkan -> tepi tajam; DPI efektif dilaporkan)
# ---------------------------------------------------------------------------
def render_png(L, path, dpi):
    cell_px = max(1, int(round(L["cell"] * dpi / 25.4)))
    eff_dpi = cell_px * 25.4 / L["cell"]
    s = eff_dpi / 25.4                                    # piksel per mm
    Wp, Hp = int(round(L["page_w"] * s)), int(round(L["page_h"] * s))
    if Wp * Hp > MAX_PNG_PIXELS:
        raise ValueError(f"PNG akan berukuran {Wp}x{Hp} piksel (terlalu besar). "
                         f"Pakai --format pdf/svg, atau turunkan --dpi.")
    page = np.full((Hp, Wp), 255, np.uint8)
    X0, Y0 = int(round(L["x0"] * s)), int(round(L["y0"] * s))
    for (cx, cy) in L["cells"]:
        x, y = X0 + (cx + 1) * cell_px, Y0 + (cy + 1) * cell_px
        page[y:y + cell_px, x:x + cell_px] = 0
    for (bx, by, bw, bh) in L["bars"]:
        x1, y1 = int(round(bx * s)), int(round(by * s))
        page[y1:y1 + max(1, int(round(bh * s))), x1:x1 + max(1, int(round(bw * s)))] = 0
    f = cv2.FONT_HERSHEY_SIMPLEX
    for (tx, ty, fs, bold, txt) in L["texts"]:
        scale = (fs * s) / 29.0
        thick = max(1, int(round(scale * (2.4 if bold else 1.6))))
        avail = Wp * 0.94
        while cv2.getTextSize(txt, f, scale, thick)[0][0] > avail and scale > 0.1:
            scale *= 0.95
        tw = cv2.getTextSize(txt, f, scale, thick)[0][0]
        cv2.putText(page, txt, (int(tx * s - tw / 2), int(ty * s)), f, scale, 0, thick, cv2.LINE_AA)
    try:
        from PIL import Image
        Image.fromarray(page).save(path, dpi=(eff_dpi, eff_dpi))
    except ImportError:
        cv2.imwrite(path, page)
    return Wp, Hp, eff_dpi


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Buat marker ArUco berukuran bebas (kamus DICT_4X4_1000)")
    ap.add_argument("--ids", type=int, nargs="+", required=True, help="ID marker (0-999)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--kotak-mm", type=float, nargs="+", help="lebar SATU kotak (mm)")
    g.add_argument("--sisi-cm", type=float, nargs="+", help="sisi bagian HITAM marker (cm)")
    g.add_argument("--total-cm", type=float, nargs="+", help="lebar total termasuk tepi putih (cm)")
    ap.add_argument("--kertas", default="auto", choices=["auto"] + list(PAPERS),
                    help="auto = ukuran halaman menyesuaikan marker (default)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "svg", "png"])
    ap.add_argument("--out", help="nama file keluaran (default aruco.<format>)")
    ap.add_argument("--dpi", type=int, default=300, help="hanya untuk --format png")
    ap.add_argument("--tanpa-label", action="store_true", help="hanya marker, tanpa teks & garis skala")
    ap.add_argument("--margin-mm", type=float, help="margin putih di sekeliling (mm)")
    a = ap.parse_args()

    for mid in a.ids:
        if not 0 <= mid <= 999:
            sys.exit(f"ID {mid} di luar rentang 0-999 (kamus {ARUCO_DICT_NAME})")

    if a.kotak_mm:
        cells, conv = a.kotak_mm, 1.0
    elif a.sisi_cm:
        cells, conv = a.sisi_cm, 10.0 / CELLS_BLACK
    else:
        cells, conv = a.total_cm, 10.0 / CELLS_TOTAL
    if len(cells) == 1:
        cells = cells * len(a.ids)
    if len(cells) != len(a.ids):
        sys.exit("Jumlah nilai ukuran harus 1 (dipakai semua ID) atau sama dengan jumlah --ids")
    cells = [v * conv for v in cells]
    if any(v <= 0 for v in cells):
        sys.exit("Ukuran harus lebih dari 0")

    pages = []
    for mid, cmm in zip(a.ids, cells):
        try:
            L = layout(mid, cmm, a.kertas, not a.tanpa_label, a.margin_mm)
        except ValueError as e:
            sys.exit(str(e))
        if a.format == "pdf" and max(L["page_w"], L["page_h"]) > MAX_PDF_MM:
            sys.exit(f"Halaman {max(L['page_w'], L['page_h']) / 10:.0f} cm melebihi batas PDF (~500 cm). "
                     f"Pakai --format svg, atau buat setengah ukuran lalu minta percetakan mencetak 200%.")
        print(f"ID {mid}: kotak {cmm:.2f} mm | sisi hitam {6 * cmm / 10:.2f} cm | "
              f"total {8 * cmm / 10:.2f} cm | halaman {L['page_w'] / 10:.1f} x {L['page_h'] / 10:.1f} cm")
        pages.append((mid, L))

    base, ext = os.path.splitext(a.out or f"aruco.{a.format}")
    ext = ext or f".{a.format}"
    if a.format == "pdf":
        out = base + ext
        render_pdf([L for _, L in pages], out)
        print(f"Tersimpan: {out} ({len(pages)} halaman). Cetak di 100% (Actual size).")
    else:
        for mid, L in pages:
            out = f"{base}{ext}" if len(pages) == 1 else f"{base}_{mid}{ext}"
            if a.format == "svg":
                render_svg(L, out)
                print(f"Tersimpan: {out}")
            else:
                try:
                    Wp, Hp, eff = render_png(L, out, a.dpi)
                except ValueError as e:
                    sys.exit(str(e))
                print(f"Tersimpan: {out} ({Wp}x{Hp} px, DPI efektif {eff:.1f}) — cetak di 100%/DPI aslinya.")


if __name__ == "__main__":
    main()