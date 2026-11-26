"""Pipeline completo: segmentação da placa + pré-processamento + OCR com EASYOCR.

Uso:  python run_easyocr.py [--imagens imagens] [--gpu]
Saída: resultados/easyocr/resultados_easyocr.txt  (texto final por imagem)
(a 1ª execução baixa os modelos do EasyOCR; precisa de internet)
"""
import argparse
import sys

from src.ocr_engines import EasyOCREngine
from src.pipeline import run_pipeline

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", default="imagens", help="pasta com as imagens")
    ap.add_argument("--gpu", action="store_true", help="usa GPU (CUDA) se disponível")
    a = ap.parse_args()
    run_pipeline(EasyOCREngine(gpu=a.gpu), "easyocr", images_dir=a.imagens)
