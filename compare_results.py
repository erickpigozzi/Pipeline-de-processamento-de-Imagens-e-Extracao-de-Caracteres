"""
Compara os resultados dos pipelines (Tesseract x EasyOCR) com o gabarito.

Uso:  python compare_results.py
Lê  : gabarito.csv  e  resultados/<motor>/detalhes_<motor>.json
Gera: resultados/comparacao.md  (tabelas)  e  resultados/comparacao_variantes.png (gráfico)
Funciona com apenas um dos motores disponíveis (compara o que existir).
"""
import csv
import json
import os
import sys

from src.postprocess import clean, apply_format, vote, LETTER_TO_DIGIT_GENERIC

ENGINES = [("tesseract", "Tesseract"), ("easyocr", "EasyOCR")]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(preds, gt):
    """preds/gt: dict imagem -> texto. Retorna acerto exato, acurácia de caracteres e CER."""
    n = len(gt)
    exact = sum(preds.get(k, "") == v for k, v in gt.items())
    total_chars = sum(len(v) for v in gt.values())
    pos_ok = sum(sum(a == b for a, b in zip(preds.get(k, ""), v)) for k, v in gt.items())
    edits = sum(levenshtein(preds.get(k, ""), v) for k, v in gt.items())
    return dict(exact=exact, n=n, pos_acc=pos_ok / total_chars, cer=edits / total_chars)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # evita erro de acentos/símbolos no Windows
    gt = {r["imagem"]: r["placa"] for r in csv.DictReader(open("gabarito.csv", encoding="utf-8"))}
    data = {}
    for slug, label in ENGINES:
        p = os.path.join("resultados", slug, f"detalhes_{slug}.json")
        if os.path.exists(p):
            data[label] = json.load(open(p, encoding="utf-8"))
        else:
            print(f"(aviso) {p} não encontrado - rode python run_{slug}.py para incluir {label}")
    if not data:
        return

    md = ["# Comparação de OCR nas placas (20 imagens)\n",
          "Gabarito: `gabarito.csv` (transcrição manual). "
          "Métricas: **acerto exato** = placa inteira correta; **acurácia por posição** = % de caracteres "
          "corretos na posição; **CER** = taxa de erro de caracteres (distância de Levenshtein / nº de caracteres; menor é melhor).\n"]

    # ---- 1) pipeline final (votação entre variantes) ----
    md += ["\n## 1. Resultado final do pipeline (votação entre variantes + formato Mercosul)\n",
           "| Motor | Acerto exato | Acurácia por posição | CER | Formato válido | Tempo total |",
           "|---|---|---|---|---|---|"]
    finals = {}
    for label, d in data.items():
        preds = {k: v["final"] for k, v in d["resultados"].items()}
        finals[label] = preds
        m = metrics(preds, gt)
        valid = sum(bool(v.get("valid")) for v in d["resultados"].values())
        md.append(f"| {label} | {m['exact']}/{m['n']} ({100*m['exact']/m['n']:.0f}%) | "
                  f"{100*m['pos_acc']:.1f}% | {100*m['cer']:.1f}% | {valid}/{m['n']} | {d['tempo_s']:.1f} s |")

    # ---- 2) por variante de pré-processamento ----
    variant_names = list(next(iter(next(iter(data.values()))["resultados"].values()))["variants"].keys())
    md += ["\n## 2. Efeito do pré-processamento (acerto exato por variante, sem votação)\n",
           "Texto **bruto** = saída do OCR limpa (só A-Z/0-9), sem corrigir pelo formato. "
           "Texto **formatado** = após correção LLLNLNN.\n",
           "| Variante | " + " | ".join(f"{l} bruto | {l} formatado" for l in data) + " |",
           "|---|" + "---|---|" * len(data)]
    chart = {l: [] for l in data}
    for v in variant_names:
        row = f"| {v} |"
        for label, d in data.items():
            raw = {k: clean(r["variants"][v]["raw"]) for k, r in d["resultados"].items() if v in r["variants"]}
            fmt = {k: r["variants"][v]["formatted"] for k, r in d["resultados"].items() if v in r["variants"]}
            mr, mf = metrics(raw, gt), metrics(fmt, gt)
            row += f" {mr['exact']}/20 | {mf['exact']}/20 |"
            chart[label].append(mf["exact"])
        md.append(row)

    # ---- 2b) sensibilidade ao pós-processamento ----
    md += ["\n## 2b. Sensibilidade ao pós-processamento (mesmas saídas brutas do OCR)\n",
           "| Motor | Só texto bruto limpo (sem formato) | Formato com tabela genérica (G→6) | Formato ajustado à fonte (G→0) |",
           "|---|---|---|---|"]
    for label, d in data.items():
        raw_v, gen_v, tun_v = {}, {}, {}
        for k, r in d["resultados"].items():
            vs = r["variants"].values()
            # bruto: variante com maior confiança, texto só limpo
            best = max(vs, key=lambda x: x["conf"]) if vs else {"raw": ""}
            raw_v[k] = clean(best["raw"])
            gen_v[k] = vote([(apply_format(x["raw"], LETTER_TO_DIGIT_GENERIC), x["conf"]) for x in vs])
            tun_v[k] = vote([(apply_format(x["raw"]), x["conf"]) for x in vs])
        a, b, c = metrics(raw_v, gt), metrics(gen_v, gt), metrics(tun_v, gt)
        md.append(f"| {label} | {a['exact']}/20 ({100*a['pos_acc']:.1f}% pos.) | "
                  f"{b['exact']}/20 ({100*b['pos_acc']:.1f}% pos.) | {c['exact']}/20 ({100*c['pos_acc']:.1f}% pos.) |")

    # ---- 3) por imagem ----
    md += ["\n## 3. Resultado por imagem (✔ = correto)\n",
           "| Imagem | Gabarito | " + " | ".join(data) + " |", "|---|---|" + "---|" * len(data)]
    for k, g in gt.items():
        cells = []
        for label in data:
            p = finals[label].get(k, "")
            cells.append(f"{p} {'✔' if p == g else '✘'}")
        md.append(f"| {k} | {g} | " + " | ".join(cells) + " |")

    # ---- 4) erros típicos ----
    md += ["\n## 4. Erros de caractere mais frequentes (gabarito → lido), pipeline final\n"]
    for label in data:
        errs = {}
        for k, g in gt.items():
            p = finals[label].get(k, "")
            if len(p) == len(g):
                for a, b in zip(g, p):
                    if a != b:
                        errs[(a, b)] = errs.get((a, b), 0) + 1
        top = sorted(errs.items(), key=lambda kv: -kv[1])[:8]
        md.append(f"- **{label}**: " + (", ".join(f"`{a}`→`{b}` ({n}x)" for (a, b), n in top) or "nenhum"))

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/comparacao.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- gráfico ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(variant_names))
        w = 0.8 / len(data)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, (label, vals) in enumerate(chart.items()):
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=25, ha="right")
        ax.set_ylabel("Placas corretas (de 20)")
        ax.set_title("Acerto exato por variante de pré-processamento")
        ax.set_ylim(0, 20)
        ax.legend()
        plt.tight_layout()
        plt.savefig("resultados/comparacao_variantes.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
