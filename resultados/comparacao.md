# Comparação de OCR nas placas (20 imagens)

Gabarito: `gabarito.csv` (transcrição manual). Métricas: **acerto exato** = placa inteira correta; **acurácia por posição** = % de caracteres corretos na posição; **CER** = taxa de erro de caracteres (distância de Levenshtein / nº de caracteres; menor é melhor).


## 1. Resultado final do pipeline (votação entre variantes + formato Mercosul)

| Motor | Acerto exato | Acurácia por posição | CER | Formato válido | Tempo total |
|---|---|---|---|---|---|
| Tesseract | 18/20 (90%) | 98.6% | 1.4% | 20/20 | 43.1 s |
| EasyOCR | 16/20 (80%) | 97.1% | 2.9% | 20/20 | 64.1 s |

## 2. Efeito do pré-processamento (acerto exato por variante, sem votação)

Texto **bruto** = saída do OCR limpa (só A-Z/0-9), sem corrigir pelo formato. Texto **formatado** = após correção LLLNLNN.

| Variante | Tesseract bruto | Tesseract formatado | EasyOCR bruto | EasyOCR formatado |
|---|---|---|---|---|
| gray | 3/20 | 7/20 | 4/20 | 8/20 |
| clahe | 1/20 | 4/20 | 7/20 | 9/20 |
| otsu | 5/20 | 12/20 | 7/20 | 11/20 |
| adaptive | 8/20 | 13/20 | 8/20 | 14/20 |
| bgnorm_otsu | 7/20 | 16/20 | 8/20 | 15/20 |
| bgnorm_median | 11/20 | 17/20 | 4/20 | 10/20 |
| bgnorm_close | 9/20 | 16/20 | 8/20 | 14/20 |

## 2b. Sensibilidade ao pós-processamento (mesmas saídas brutas do OCR)

| Motor | Só texto bruto limpo (sem formato) | Formato com tabela genérica (G→6) | Formato ajustado à fonte (G→0) |
|---|---|---|---|
| Tesseract | 10/20 (89.3% pos.) | 14/20 (95.0% pos.) | 18/20 (98.6% pos.) |
| EasyOCR | 7/20 (83.6% pos.) | 16/20 (97.1% pos.) | 16/20 (97.1% pos.) |

## 3. Resultado por imagem (✔ = correto)

| Imagem | Gabarito | Tesseract | EasyOCR |
|---|---|---|---|
| 1.JPG | IFA1T95 | IFA1T95 ✔ | IFA1T95 ✔ |
| 2.JPG | NEV1C08 | NEV1C08 ✔ | NEV1C08 ✔ |
| 3.JPG | IYZ2Z18 | LYZ2Z18 ✘ | IYZ2Z18 ✔ |
| 4.JPG | QOW9N47 | QOW9N47 ✔ | QOW9N47 ✔ |
| 5.JPG | USQ0T32 | USQ0T32 ✔ | USQ0T32 ✔ |
| 6.JPG | TYL3X75 | TYL3X75 ✔ | TYL3X75 ✔ |
| 7.JPG | GXQ2J95 | GXQ2I95 ✘ | GXO2J95 ✘ |
| 8.JPG | XHY6U29 | XHY6U29 ✔ | XHY6U29 ✔ |
| 9.JPG | AOA4J16 | AOA4J16 ✔ | AOA4J16 ✔ |
| 10.JPG | PDU7G22 | PDU7G22 ✔ | PDU7G22 ✔ |
| 11.JPG | MNL3U98 | MNL3U98 ✔ | MNL3U98 ✔ |
| 12.JPG | JET7G88 | JET7G88 ✔ | JET7G88 ✔ |
| 13.JPG | ABP8Q89 | ABP8Q89 ✔ | ABP8O89 ✘ |
| 14.JPG | PHI8H00 | PHI8H00 ✔ | PHI8H00 ✔ |
| 15.JPG | THY7E19 | THY7E19 ✔ | THY2E19 ✘ |
| 16.JPG | UHK5O55 | UHK5O55 ✔ | UHK5O55 ✔ |
| 17.JPG | NZQ3M77 | NZQ3M77 ✔ | NZQ3M77 ✔ |
| 18.JPG | XDA5E50 | XDA5E50 ✔ | XDA5E50 ✔ |
| 19.JPG | ESI4I93 | ESI4I93 ✔ | ESI4I93 ✔ |
| 20.JPG | TGG1D44 | TGG1D44 ✔ | TGG1O44 ✘ |

## 4. Erros de caractere mais frequentes (gabarito → lido), pipeline final

- **Tesseract**: `I`→`L` (1x), `J`→`I` (1x)
- **EasyOCR**: `Q`→`O` (2x), `7`→`2` (1x), `D`→`O` (1x)
