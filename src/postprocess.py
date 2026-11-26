"""
Etapa 4 - Pós-processamento do texto reconhecido.

As placas do dataset seguem o padrão Mercosul brasileiro: LLLNLNN
(3 letras, 1 número, 1 letra, 2 números). Conhecer o padrão permite:
  * limpar o texto (apenas A-Z e 0-9, maiúsculas);
  * escolher a janela de 7 caracteres mais compatível quando o OCR devolve lixo extra;
  * corrigir confusões clássicas por posição (ex.: '5' <-> 'S', '0' <-> 'O').
Também implementa a votação ponderada entre as variantes de pré-processamento.
"""
import re
from collections import defaultdict
from itertools import combinations

PATTERN = "LLLNLNN"

# Confusões visuais típicas de OCR em placas
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
                   "5": "S", "6": "G", "7": "T", "8": "B"}

# Tabela GENÉRICA (letra lida numa posição que deve ser dígito -> dígito parecido)
LETTER_TO_DIGIT_GENERIC = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                           "S": "5", "B": "8", "G": "6", "T": "7", "A": "4", "E": "3"}

# Tabela AJUSTADA À FONTE do dataset: o dígito zero é cortado por uma barra
# diagonal e os OCRs o leem como "G". Em posição numérica, "G" => "0".
# (Ajuste feito após observar os erros nestas 20 imagens; ver README, seção
# "Limitações". A comparação mostra o resultado com as duas tabelas.)
LETTER_TO_DIGIT = {**LETTER_TO_DIGIT_GENERIC, "G": "0"}


def clean(text):
    """Maiúsculas e somente A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _fits(ch, kind):
    return ch.isalpha() if kind == "L" else ch.isdigit()


def _best_subsequence(text):
    """
    Se o OCR devolveu mais de 7 caracteres (ruído / caractere duplicado), escolhe
    a subsequência de 7 que melhor respeita o padrão LLLNLNN.
    Desempate: prefere remover caracteres nas bordas (onde costuma haver lixo,
    ex.: resto do rótulo "BR") e depois os mais à esquerda.
    """
    n = len(text)
    if n <= 7:
        return text
    best, best_key = text[:7], None
    for keep in combinations(range(n), 7):
        fit = sum(_fits(text[i], k) for i, k in zip(keep, PATTERN))
        removed = [i for i in range(n) if i not in keep]
        edge = sum(1 for i in removed if i in (0, n - 1))
        key = (fit, edge, -sum(removed))
        if best_key is None or key > best_key:
            best, best_key = "".join(text[i] for i in keep), key
    return best


def apply_format(text, letter_to_digit=None):
    """Limpa e corrige o texto conforme o padrão LLLNLNN."""
    l2d = letter_to_digit or LETTER_TO_DIGIT
    t = _best_subsequence(clean(text))
    out = []
    for ch, kind in zip(t, PATTERN):
        if _fits(ch, kind):
            out.append(ch)
        elif kind == "L":
            out.append(DIGIT_TO_LETTER.get(ch, ch))
        else:
            out.append(l2d.get(ch, ch))
    return "".join(out) + t[7:]


def is_valid(text):
    return re.fullmatch(r"[A-Z]{3}[0-9][A-Z][0-9]{2}", text or "") is not None


def vote(candidates):
    """
    Votação por posição entre as variantes.
    candidates: lista de (texto_já_formatado, confiança). Só participam textos
    com 7 caracteres. O peso de cada voto é 0.1 + confiança do OCR.
    Se nenhum candidato tiver 7 caracteres, devolve o mais confiante.
    """
    full = [(t, 0.1 + c) for t, c in candidates if len(t) == 7]  # 0.1 evita peso nulo
    if not full:
        cands = [(t, c) for t, c in candidates if t]
        return max(cands, key=lambda x: x[1])[0] if cands else ""
    out = []
    for pos in range(7):
        score = defaultdict(float)
        for t, c in full:
            score[t[pos]] += c
        out.append(max(score.items(), key=lambda kv: kv[1])[0])
    return "".join(out)
