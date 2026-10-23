# /// script
# requires-python = ">=3.12"
# dependencies = ["qrcode==8.2", "reportlab==4.4.3"]
# ///
"""Generate the workshop's direct GitHub QR code as SVG and vector PDF.

Run from any directory with:
    uv run /path/to/scripts/generate-workshop-qr.py
"""

from pathlib import Path

import qrcode
from qrcode.image.svg import SvgPathFillImage
from reportlab.pdfgen import canvas

WORKSHOP_URL = "https://github.com/uashogeschoolutrecht/IMAGINE-workshop-2026"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "presentation" / "images"
MODULE_SIZE = 10
QUIET_ZONE = 4


def main() -> None:
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=MODULE_SIZE,
        border=QUIET_ZONE,
    )
    qr.add_data(WORKSHOP_URL)
    qr.make(fit=True)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = IMAGE_DIR / "github-link.svg"
    qr.make_image(
        image_factory=SvgPathFillImage,
        attrib={"role": "img", "aria-label": f"QR code for {WORKSHOP_URL}"},
    ).save(svg_path)

    # The same matrix, including the quiet zone, supplies the pdflatex asset.
    matrix = qr.get_matrix()
    side = len(matrix) * MODULE_SIZE
    pdf_path = IMAGE_DIR / "github-link.pdf"
    pdf = canvas.Canvas(
        str(pdf_path), pagesize=(side, side), pageCompression=1, invariant=1
    )
    pdf.setTitle("Workshop download QR code")
    pdf.setSubject(WORKSHOP_URL)
    pdf.setCreator("scripts/generate-workshop-qr.py")
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, side, side, fill=1, stroke=0)
    pdf.setFillColorRGB(0, 0, 0)
    modules = pdf.beginPath()
    for row, cells in enumerate(matrix):
        for column, dark in enumerate(cells):
            if dark:
                # PDF coordinates start at the bottom left; QR rows at the top.
                modules.rect(
                    column * MODULE_SIZE,
                    side - (row + 1) * MODULE_SIZE,
                    MODULE_SIZE,
                    MODULE_SIZE,
                )
    pdf.drawPath(modules, fill=1, stroke=0)
    pdf.showPage()
    pdf.save()
    print(f"Generated {svg_path} and {pdf_path}")
    print(f"QR destination: {WORKSHOP_URL}")


if __name__ == "__main__":
    main()
