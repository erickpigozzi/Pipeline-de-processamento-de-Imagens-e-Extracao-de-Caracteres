"""
Ponto de entrada do projeto: roda os DOIS pipelines (Tesseract e EasyOCR) sobre as
mesmas imagens, com a mesma segmentação e o mesmo pré-processamento, e gera a comparação.

    python main.py

Se o EasyOCR não estiver instalado, apenas o pipeline do Tesseract é executado.
Cada pipeline também pode ser executado separadamente:
    python run_tesseract.py
    python run_easyocr.py
"""
import sys

from src.ocr_engines import TesseractEngine
from src.pipeline import run_pipeline
import compare_results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Pipeline 1: Tesseract ===")
    run_pipeline(TesseractEngine(psm=7), "tesseract")

    print("\n=== Pipeline 2: EasyOCR ===")
    try:
        from src.ocr_engines import EasyOCREngine
        run_pipeline(EasyOCREngine(gpu=False), "easyocr")
    except ImportError:
        print("EasyOCR não instalado (pip install easyocr) - pipeline ignorado.")

    print("\n=== Comparação com o gabarito ===")
    compare_results.main()


if __name__ == "__main__":
    main()
