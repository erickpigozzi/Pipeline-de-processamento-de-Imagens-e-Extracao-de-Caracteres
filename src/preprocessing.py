"""
Etapa 2 - Pré-processamento da região dos caracteres para OCR.

Desafios do dataset (observados nas imagens):
  * iluminação desigual / degradê de desfoque sobre a placa;
  * texto cinza-claro sobre fundo branco (baixo contraste, ex.: imagens 5, 8, 18);
  * texto escuro sobre fundo cinza (ex.: imagens 6, 20);
  * marca d'água diagonal "MERCOSUL BRASIL" (hachura) dentro dos caracteres.

Cada função abaixo implementa uma técnica diferente. A função `build_variants`
gera todas as variantes; o MESMO conjunto é usado nos dois pipelines
(Tesseract e EasyOCR), garantindo uma comparação justa entre as ferramentas.
"""
import cv2
import numpy as np

TARGET_H = 48  # altura padrão (px) do ROI antes do OCR


def to_gray_resized(roi_bgr, target_h=TARGET_H):
    """BGR -> cinza e redimensionamento para altura fixa (mantém proporção)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    s = target_h / gray.shape[0]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=interp)


def background_normalize(gray):
    """
    Normalização de iluminação: o fechamento morfológico (closing) com elemento
    grande "apaga" os caracteres escuros e estima o fundo; dividir a imagem por
    esse fundo remove degradês e realça o contraste do texto.
    """
    k = int(gray.shape[0] * 0.9) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def otsu(gray, blur=3):
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur else gray
    _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def pad_border(img, frac=0.25):
    """Borda ao redor do texto (OCRs funcionam melhor com margem)."""
    pad = int(img.shape[0] * frac)
    val = 255 if img.mean() > 127 else 0
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=val)


# --------------------------------------------------------------------------- #
# Variantes de pré-processamento
# --------------------------------------------------------------------------- #
def v_gray(roi):
    """Baseline: apenas escala de cinza + redimensionamento."""
    return to_gray_resized(roi)


def v_clahe(roi):
    """Equalização adaptativa de histograma (CLAHE) para realçar contraste local."""
    g = to_gray_resized(roi)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(g)


def v_otsu(roi):
    """Binarização global de Otsu."""
    return otsu(to_gray_resized(roi))


def v_adaptive(roi):
    """Limiarização adaptativa gaussiana (limiar local por vizinhança)."""
    g = to_gray_resized(roi)
    block = int(g.shape[0] * 0.8) | 1
    return cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, 10)


def v_bgnorm_otsu(roi):
    """Normalização de fundo + Otsu."""
    return otsu(background_normalize(to_gray_resized(roi)))


def v_bgnorm_median(roi):
    """Normalização de fundo + filtro de mediana (suaviza a hachura) + Otsu."""
    g = cv2.medianBlur(background_normalize(to_gray_resized(roi)), 5)
    return otsu(g, blur=0)


def v_bgnorm_close(roi):
    """
    Normalização de fundo + Otsu + fechamento morfológico dos caracteres
    (preenche as falhas brancas causadas pela marca d'água).
    """
    b = otsu(background_normalize(to_gray_resized(roi)))
    return 255 - cv2.morphologyEx(255 - b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


VARIANTS = {
    "gray": v_gray,
    "clahe": v_clahe,
    "otsu": v_otsu,
    "adaptive": v_adaptive,
    "bgnorm_otsu": v_bgnorm_otsu,
    "bgnorm_median": v_bgnorm_median,
    "bgnorm_close": v_bgnorm_close,
}


def build_variants(roi_bgr):
    """Retorna {nome: imagem pré-processada (com borda)} para todas as variantes."""
    return {name: pad_border(fn(roi_bgr)) for name, fn in VARIANTS.items()}
