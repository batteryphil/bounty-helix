import os
import subprocess

def pdftotext(pdf_path, txt_path, options={}):
    # Run the pdftotext command
    cmd = ["pdftotext"] + list(options) + [pdf_path, txt_path]
    subprocess.run(cmd, check=True)

    print(f"Converted {pdf_path} to {txt_path}")
