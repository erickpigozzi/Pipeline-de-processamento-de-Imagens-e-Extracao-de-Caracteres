"""
Etapa 3 - Motores de OCR.

Ambos os motores expõem a mesma interface:  engine.recognize(img) -> (texto, confiança 0-1)
Assim, o restante do pipeline (segmentação, pré-processamento, pós-processamento e
votação) é IDÊNTICO e a única diferença entre os experimentos é a ferramenta de OCR.
"""
import os
import shutil

import cv2

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def find_tesseract():
    """Localiza o executável do Tesseract (PATH ou instalações padrão do Windows)."""
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
              "/usr/bin/tesseract", "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Tesseract não encontrado. Instale o Tesseract OCR e/ou ajuste TESSERACT_CMD "
        "(ex.: C:\\Program Files\\Tesseract-OCR\\tesseract.exe).")


class TesseractEngine:
    """Tesseract via pytesseract (requer o binário `tesseract` instalado no sistema)."""
    name = "Tesseract"

    def __init__(self, psm=7):
        import pytesseract
        self.pt = pytesseract
        # variável de ambiente TESSERACT_CMD tem prioridade sobre a autodetecção
        self.pt.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD") or find_tesseract()
        # --psm 7: imagem = uma única linha de texto
        # --oem 3: motor padrão (LSTM)
        # whitelist: só letras maiúsculas e dígitos
        # dawgs desligados: placas não são palavras de dicionário
        self.config = (f"--oem 3 --psm {psm} -c tessedit_char_whitelist={ALLOWLIST} "
                       "-c load_system_dawg=0 -c load_freq_dawg=0")

    def recognize(self, img):
        try:
            d = self.pt.image_to_data(img, config=self.config,
                                      output_type=self.pt.Output.DICT)
        except self.pt.TesseractError:
            return "", 0.0
        words, confs = [], []
        for t, c in zip(d["text"], d["conf"]):
            if t.strip() and float(c) >= 0:
                words.append(t.strip())
                confs.append(float(c) / 100.0)
        if not words:
            return "", 0.0
        return "".join(words), sum(confs) / len(confs)


class EasyOCREngine:
    """
    EasyOCR (CRAFT + CRNN). Na primeira execução baixa os modelos (~100 MB).

    Segue o uso dos exemplos da disciplina: `readtext(..., allowlist=...)`.
    Se o detector (CRAFT) não encontrar texto no recorte, usa `recognize` com a
    imagem inteira como caixa de texto (pula a detecção) como alternativa.
    """
    name = "EasyOCR"

    def __init__(self, gpu=False):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    @staticmethod
    def _join(res):
        """Ordena da esquerda para a direita e concatena textos/confianças."""
        res = [r for r in res if str(r[1]).strip()]
        if not res:
            return "", 0.0
        res = sorted(res, key=lambda r: min(p[0] for p in r[0]))
        return "".join(str(r[1]).strip() for r in res), sum(float(r[2]) for r in res) / len(res)

    def recognize(self, img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            text, conf = self._join(self.reader.readtext(
                gray, detail=1, paragraph=False, allowlist=ALLOWLIST))
        except Exception:
            text, conf = "", 0.0
        if len(text) >= 5:
            return text, conf
        h, w = gray.shape[:2]
        try:  # alternativa: sem detecção, a linha inteira é uma caixa de texto
            t2, c2 = self._join(self.reader.recognize(
                gray, horizontal_list=[[0, w, 0, h]], free_list=[],
                allowlist=ALLOWLIST, detail=1, paragraph=False))
        except Exception:
            t2, c2 = "", 0.0
        return (t2, c2) if len(t2) > len(text) else (text, conf)
