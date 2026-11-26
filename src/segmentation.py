"""
Etapa 1 - Detecção, segmentação e extração da placa veicular.

Estratégia (Processamento de Imagens clássico, sem deep learning):

1. Conversão BGR -> HSV e limiarização da cor AZUL. A placa Mercosul possui uma
   faixa azul no topo, que é a característica visual mais estável do dataset.
2. Operações morfológicas (fechamento + abertura) para unir fragmentos da faixa
   e remover ruído, seguidas de busca de contornos.
3. Cada contorno vira um CANDIDATO e recebe um score de verificação do "modelo
   de placa": proporção da faixa, preenchimento, bandeira verde/amarela no canto
   direito, corpo claro/pouco saturado logo abaixo e presença de texto escuro.
   Isso elimina falsos positivos (carroceria azul, céu, adesivos, placa pequena
   do estepe etc.).
4. Com a faixa escolhida, usa-se cv2.minAreaRect para obter a inclinação e uma
   transformação de perspectiva (homografia) para "endireitar" a placa.
5. A borda inferior é localizada pela moldura escura tracejada da placa; assim
   o recorte não inclui a placa original do veículo que aparece por baixo.
6. Saídas: placa completa retificada e ROI apenas dos caracteres.
"""
import cv2
import numpy as np

W_WARP = 900        # largura padrão da placa retificada (px)
K_GENEROUS = 2.6    # razão largura/altura da região "generosa" usada para achar a borda inferior


# --------------------------------------------------------------------------- #
# 1-3. Detecção da faixa azul e ranqueamento dos candidatos
# --------------------------------------------------------------------------- #
def _blue_green_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Azul da faixa (H ~ 100-130 no OpenCV). V mínimo baixo porque em fotos
    # escuras (ex.: imagem 20) o azul da faixa fica bem escuro.
    blue = cv2.inRange(hsv, (100, 90, 40), (130, 255, 255))
    # Verde/amarelo da bandeira do Brasil
    green = cv2.inRange(hsv, (30, 70, 70), (85, 255, 255))
    return hsv, blue, green


def find_plate_candidates(img):
    """Retorna ([(score, contorno, (x,y,w,h))] ordenado por score, máscara azul)."""
    H, W = img.shape[:2]
    hsv, blue, green = _blue_green_masks(img)

    # Fechamento horizontal une trechos da faixa; abertura remove pontos soltos.
    m = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # A faixa é um retângulo fino e comprido: filtros geométricos
        if w < 70 or h < 5 or not (5 <= w / h <= 32) or w > 0.95 * W:
            continue
        fill = cv2.countNonZero(m[y:y + h, x:x + w]) / float(w * h)
        if fill < 0.3:
            continue

        # Bandeira (verde/amarelo) na extremidade direita da faixa
        gx0, gx1 = x + int(w * .8), min(W, x + w + int(w * .06))
        gy0, gy1 = max(0, y - int(h * .3)), min(H, y + int(h * 1.3))
        flag = (green[gy0:gy1, gx0:gx1] > 0).mean()

        # Corpo da placa logo abaixo da faixa: claro, pouco saturado, com texto escuro
        ph = int(w / 3.6)
        body = hsv[y + h:min(H, y + ph), x:x + w]
        if body.shape[0] < 5:
            continue
        lowsat = (body[..., 1] < 80).mean()
        bright = (body[..., 2] > 110).mean()
        dark = (body[..., 2] < 80).mean()

        score = (fill + min(flag * 15, 1.5) + lowsat + bright
                 + (0.5 if 0.08 < dark < 0.5 else 0.0)
                 + 2.0 * w / W)  # prefere a placa maior (a principal da cena)
        cands.append((score, c, (x, y, w, h)))
    return sorted(cands, key=lambda t: -t[0]), m


# --------------------------------------------------------------------------- #
# 4. Retificação (perspectiva) a partir da faixa azul
# --------------------------------------------------------------------------- #
def _order_box(box):
    s = box.sum(1)
    d = np.diff(box, axis=1)[:, 0]
    tl, br = box[np.argmin(s)], box[np.argmax(s)]
    tr, bl = box[np.argmin(d)], box[np.argmax(d)]
    return tl, tr, br, bl


def _warp_below_band(img, contour, k=K_GENEROUS, w0=W_WARP):
    """
    Retifica a região da faixa azul + área abaixo dela (altura = largura/k).
    Retorna (imagem retificada, altura da faixa azul já na escala retificada).
    """
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(contour)
    tl, tr, br, bl = _order_box(cv2.boxPoints(((cx, cy), (rw, rh), ang)))
    top = np.linalg.norm(tr - tl)
    band_h = np.linalg.norm(bl - tl)
    v = (tr - tl) / top
    n = np.array([-v[1], v[0]])  # normal à faixa
    if n[1] < 0:
        n = -n                   # garante que aponta para baixo
    hh = top / k
    src = np.float32([tl, tr, tr + n * hh, tl + n * hh])
    dst = np.float32([[0, 0], [w0, 0], [w0, w0 / k], [0, w0 / k]])
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (w0, int(w0 / k)), flags=cv2.INTER_CUBIC)
    return warp, band_h * (w0 / top)


# --------------------------------------------------------------------------- #
# 5. Borda inferior da placa (moldura tracejada escura)
# --------------------------------------------------------------------------- #
def _find_bottom(warp, band_bottom):
    """
    Localiza a moldura inferior (linha preta tracejada, ~10 px de espessura) no
    recorte retificado. A placa do veículo original pode aparecer abaixo da
    moldura; por isso procura-se a PRIMEIRA faixa de linhas quase totalmente
    escuras depois da região do texto, e não o máximo global.
    Retorna (y_inicio_da_moldura, y_fim_da_moldura).
    """
    H, W = warp.shape[:2]
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)[:, int(.10 * W):int(.95 * W)]
    # limiar adaptativo: metade do brilho típico do corpo da placa
    body = gray[band_bottom + 5:band_bottom + 60]
    thr = max(40, 0.5 * np.percentile(body, 70))
    dark_frac = (gray < thr).mean(axis=1)

    start = None
    for y in range(band_bottom + 50, H):          # texto tem altura mínima ~50 px
        if dark_frac[y] > 0.6:
            if start is None:
                start = y
        elif start is not None:
            if y - start >= 4:                    # espessura mínima da moldura
                return start, y
            start = None
    if start is not None:
        return start, H
    return int(.6 * H), int(.65 * H)              # fallback


def _band_bottom(warp):
    """
    Última linha da faixa azul no recorte retificado, medida pelo perfil de azul
    (fração de pixels azuis por linha). Mais robusto do que usar o contorno
    original, que pode estar fundido com outras regiões azuis da cena.
    """
    H, W = warp.shape[:2]
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 60, 30), (135, 255, 255))[:, int(.06 * W):int(.94 * W)]
    frac = blue.mean(axis=1) / 255.0
    lim = int(.45 * H)
    rows = np.where(frac[:lim] > 0.9)[0]
    if len(rows) == 0:
        return int(.24 * H)
    # percorre a partir da 1ª linha azul enquanto a faixa for contínua (tolera 3 linhas de falha)
    end, gap = rows[0], 0
    for y in range(rows[0], lim):
        if frac[y] > 0.9:
            end, gap = y, 0
        else:
            gap += 1
            if gap > 3:
                break
    return end


def extract_plate(img):
    """
    Pipeline de segmentação completo.
    Retorna dict com: plate (placa retificada), text_roi (só caracteres),
    box (retângulo da faixa), contour, mask. Retorna None se não achar placa.
    """
    cands, mask = find_plate_candidates(img)
    if not cands:
        return None
    score, contour, box = cands[0]
    warp, _ = _warp_below_band(img, contour)
    W = warp.shape[1]
    band_h = _band_bottom(warp)
    yb, yb_end = _find_bottom(warp, band_h)

    # Placa completa: do topo da faixa até a moldura inferior
    plate = warp[0:min(yb_end + 2, warp.shape[0])]

    # ROI dos caracteres:
    #  - topo  : logo abaixo da faixa azul (altura medida na detecção)
    #  - base  : acima da moldura tracejada inferior
    #  - esq.  : depois do rótulo "BR" (canto inferior esquerdo)
    #  - dir.  : antes da moldura tracejada lateral
    y0 = int(band_h + 2)
    y1 = yb - 2
    x0, x1 = int(.08 * W), int(.975 * W)
    text_roi = plate[y0:y1, x0:x1].copy()
    return dict(plate=plate, text_roi=text_roi, box=box, score=score,
                mask=mask, contour=contour, bottom=yb, band_h=band_h)


def draw_detection(img, res):
    """Imagem anotada com a faixa azul detectada (para relatório/depuração)."""
    out = img.copy()
    if res is None:
        return out
    x, y, w, h = res["box"]
    cv2.drawContours(out, [res["contour"]], -1, (0, 255, 0), 3)
    cv2.rectangle(out, (x, y), (x + w, y + int(w / 4.4)), (0, 0, 255), 3)
    return out
