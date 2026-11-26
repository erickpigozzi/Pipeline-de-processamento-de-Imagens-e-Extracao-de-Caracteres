"""Pipeline completo: segmentação da placa + pré-processamento + OCR com TESSERACT.

Uso:  python run_tesseract.py [--imagens imagens] [--variantes]
Saída: resultados/tesseract/resultados_tesseract.txt  (texto final por imagem)
"""
import argparse
import sys

from src.ocr_engines import TesseractEngine
from src.pipeline import run_pipeline

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", default="imagens", help="pasta com as imagens")
    ap.add_argument("--variantes", action="store_true", help="salva as imagens pré-processadas")
    a = ap.parse_args()
    run_pipeline(TesseractEngine(psm=7), "tesseract", images_dir=a.imagens, save_variants=a.variantes)
