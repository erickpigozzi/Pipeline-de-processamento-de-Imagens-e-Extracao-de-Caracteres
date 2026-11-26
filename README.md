# Comparação de OCR em Placas Veiculares (Tesseract x EasyOCR)

Projeto da disciplina de Processamento de Imagens. O pipeline segmenta e extrai placas
Mercosul de 20 imagens e aplica **dois OCRs diferentes sobre exatamente o mesmo
processamento**, para comparar o desempenho das ferramentas.

## Pipeline

```
imagem -> 1. segmentação da placa -> 2. extração/retificação -> 3. ROI dos caracteres
       -> 4. pré-processamento (7 variantes) -> 5. OCR (Tesseract | EasyOCR)
       -> 6. pós-processamento (padrão LLLNLNN) -> 7. votação entre variantes -> texto final
```

1. **Segmentação** (`src/segmentation.py`): limiarização HSV da faixa azul da placa, morfologia
   (fechamento/abertura), busca de contornos e *score* de verificação (proporção da faixa, bandeira
   verde/amarela, corpo claro com texto escuro, tamanho). Elimina falsos positivos como carroceria
   azul e a placa pequena do estepe (imagem 2).
2. **Extração**: `minAreaRect` + transformação de perspectiva endireitam a placa. A moldura
   tracejada inferior é localizada para que o recorte **não inclua a placa original do carro**
   que aparece embaixo (imagens 5, 7 e 20).
3. **ROI**: somente os caracteres (sem faixa azul, moldura nem o rótulo "BR").
4. **Pré-processamento** (`src/preprocessing.py`): `gray`, `clahe`, `otsu`, `adaptive`,
   `bgnorm_otsu`, `bgnorm_median`, `bgnorm_close` (normalização de fundo por *closing* morfológico,
   filtro de mediana, fechamento da hachura da marca d'água).
5. **OCR** (`src/ocr_engines.py`): interface idêntica para os dois motores.
6. **Pós-processamento** (`src/postprocess.py`): limpeza, escolha da melhor subsequência de 7
   caracteres e correção letra↔dígito por posição conforme o padrão Mercosul `LLLNLNN`.
7. **Votação** por posição entre as variantes, ponderada pela confiança do OCR.

## Estrutura

```
imagens/                 20 imagens originais + (gabarito.csv na raiz)
placas_segmentadas/      deteccao/, placas_extraidas/, rois_caracteres/  (evidências da segmentação)
resultados/
  tesseract/resultados_tesseract.txt          <- texto final por imagem (Tesseract)
  tesseract/resultados_tesseract_detalhado.txt   saída bruta de cada variante
  easyocr/resultados_easyocr.txt              <- gerado por run_easyocr.py
  comparacao.md / comparacao_variantes.png    tabelas e gráfico (compare_results.py)
src/                     segmentation, preprocessing, ocr_engines, postprocess, pipeline
main.py                  roda os dois pipelines + comparação
run_tesseract.py         pipeline completo com Tesseract
run_easyocr.py           pipeline completo com EasyOCR
compare_results.py       métricas contra o gabarito
```

## Instalação e execução

```bash
pip install -r requirements.txt
```
É necessário instalar o **Tesseract OCR** no computador (no Windows, o caminho padrão
`C:\Program Files\Tesseract-OCR\tesseract.exe` é detectado automaticamente; caso contrário defina a
variável de ambiente `TESSERACT_CMD`). O EasyOCR baixa seus modelos na primeira execução.

```bash
python main.py            # os dois pipelines + comparação
python run_tesseract.py   # só Tesseract
python run_easyocr.py     # só EasyOCR
python compare_results.py # (re)gera resultados/comparacao.md
```

## Resultados

Gabarito (`gabarito.csv`): transcrição manual das 20 placas.
Segmentação: **20/20 placas localizadas e extraídas corretamente**.

| Motor | Acerto exato | Acurácia por caractere | CER |
|---|---|---|---|
| Tesseract | 17/20 (85%) | 97,9% | 2,1% |
| EasyOCR | _rodar `python run_easyocr.py` e preencher_ | | |

Observações sobre o Tesseract (detalhes em `resultados/comparacao.md`):

* O pré-processamento pesa muito: sem formato, a melhor variante acerta 9/20; as variantes com
  normalização de fundo + votação + padrão Mercosul chegam a 17/20. `gray` e `clahe` foram as piores.
* O Tesseract lê o **zero com barra** da fonte da placa como "G". Em posição numérica, "G" é
  convertido em "0".
* Erros restantes: `IYZ2Z18`→`LYZ2Z18`, `GXQ2J95`→`GXQ2I95`, `MNL3U98`→`MNL5U98`.

## Limitações

* A conversão `G→0` foi definida **depois de observar os erros nestas mesmas 20 imagens**; com a
  tabela genérica (`G→6`) o Tesseract acerta 13/20. `compare_results.py` mostra as duas versões.
* O gabarito foi transcrito manualmente; confira-o.
* A segmentação usa a faixa azul/bandeira da placa Mercosul; placas antigas (cinza) exigiriam outro detector.
