import os
from pathlib import Path

import win32com.client as win32

from config import DOC_INPUT_DIR, DOCX_DIR


def convert_all_docs_to_docx():
    DOCX_DIR.mkdir(parents=True, exist_ok=True)

    word = win32.Dispatch("Word.Application")
    word.Visible = False

    try:
        for fname in os.listdir(DOC_INPUT_DIR):
            if not fname.lower().endswith(".doc") or fname.lower().endswith(".docx"):
                continue

            src = DOC_INPUT_DIR / fname
            dst = DOCX_DIR / (Path(fname).stem + ".docx")

            print(f"Converting {src.name} -> {dst.name}")
            doc = word.Documents.Open(str(src))
            doc.SaveAs(str(dst), FileFormat=16)  # 16 = docx
            doc.Close()
    finally:
        word.Quit()

    print("✔ Conversion finished.")


if __name__ == "__main__":
    convert_all_docs_to_docx()
