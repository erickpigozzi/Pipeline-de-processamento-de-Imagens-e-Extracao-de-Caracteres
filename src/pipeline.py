"""
Pipeline completo (comum aos dois motores de OCR):

    imagem -> segmentação da placa -> extração/retificação -> ROI dos caracteres
           -> pré-processamento (várias variantes) -> OCR -> pós-processamento
           -> votação entre variantes -> texto final

Gera, para cada motor:
    results/<motor>/resultados_<motor>.txt           texto final por imagem
    results/<motor>/resultados_<motor>_detalhado.txt saída bruta por variante
    results/<motor>/detalhes_<motor>.json            dados completos (usados na comparação)
"""
import glob
import json
import os
import time

import cv2

from .segmentation import extract_plate, draw_detection
from .preprocessing import build_variants
from .postprocess import apply_format, vote, is_valid


def list_images(images_dir):
    files = glob.glob(os.path.join(images_dir, "*.JPG")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    return sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def save_segmentation_outputs(img, res, name, outputs_dir):
    """Salva a detecção, a placa extraída e o ROI dos caracteres (evidência da etapa 1)."""
    stem = os.path.splitext(name)[0]
    for sub in ("deteccao", "placas_extraidas", "rois_caracteres"):
        os.makedirs(os.path.join(outputs_dir, sub), exist_ok=True)
    cv2.imwrite(os.path.join(outputs_dir, "deteccao", f"{stem}_deteccao.jpg"), draw_detection(img, res))
    cv2.imwrite(os.path.join(outputs_dir, "placas_extraidas", f"{stem}_placa.png"), res["plate"])
    cv2.imwrite(os.path.join(outputs_dir, "rois_caracteres", f"{stem}_roi.png"), res["text_roi"])


def run_pipeline(engine, slug, images_dir="imagens", results_dir="resultados",
                 outputs_dir="placas_segmentadas", save_variants=False):
    out_dir = os.path.join(results_dir, slug)
    os.makedirs(out_dir, exist_ok=True)
    details = {}
    t0 = time.time()

    for path in list_images(images_dir):
        name = os.path.basename(path)
        img = cv2.imread(path)
        res = extract_plate(img)                       # etapas 1 e 2: segmentação + extração
        if res is None:
            details[name] = dict(final="", variants={}, error="placa não detectada")
            print(f"[{engine.name}] {name}: placa NÃO detectada")
            continue
        save_segmentation_outputs(img, res, name, outputs_dir)

        variants = build_variants(res["text_roi"])     # etapa 3: pré-processamento
        per_variant, cands = {}, []
        for vname, vimg in variants.items():           # etapa 4: OCR em cada variante
            raw, conf = engine.recognize(vimg)
            fmt = apply_format(raw)                    # etapa 5: pós-processamento
            per_variant[vname] = dict(raw=raw, conf=round(conf, 4), formatted=fmt)
            cands.append((fmt, conf))
            if save_variants:
                d = os.path.join(outputs_dir, "variantes", os.path.splitext(name)[0])
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{vname}.png"), vimg)

        final = vote(cands)                            # votação ponderada entre variantes
        details[name] = dict(final=final, valid=is_valid(final), variants=per_variant)
        print(f"[{engine.name}] {name}: {final}")

    elapsed = time.time() - t0
    _write_outputs(engine, slug, details, out_dir, elapsed)
    return details


def _write_outputs(engine, slug, details, out_dir, elapsed):
    # 1) arquivo principal (entrega): texto final por imagem
    main = os.path.join(out_dir, f"resultados_{slug}.txt")
    with open(main, "w", encoding="utf-8") as f:
        f.write(f"# Texto das placas reconhecido pelo pipeline com {engine.name}\n")
        f.write("# formato: imagem: texto\n")
        for name, d in details.items():
            f.write(f"{name}: {d['final']}\n")

    # 2) arquivo detalhado: saída bruta e formatada de cada variante
    det = os.path.join(out_dir, f"resultados_{slug}_detalhado.txt")
    with open(det, "w", encoding="utf-8") as f:
        f.write(f"# Saída detalhada - {engine.name} (tempo total: {elapsed:.1f}s)\n")
        for name, d in details.items():
            f.write(f"\n== {name}  ->  FINAL: {d['final']}\n")
            for v, r in d["variants"].items():
                f.write(f"   {v:15s} bruto={r['raw']!r:14s} conf={r['conf']:.2f}  formatado={r['formatted']}\n")

    # 3) JSON completo
    with open(os.path.join(out_dir, f"detalhes_{slug}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(engine=engine.name, tempo_s=round(elapsed, 2), resultados=details),
                  f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {out_dir}/ (tempo: {elapsed:.1f}s)")
