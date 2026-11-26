# Segmentação de placas veiculares e comparação de OCR (Tesseract x EasyOCR)

Trabalho de Processamento de Imagens. Pipeline em Python/OpenCV que detecta e extrai placas
Mercosul de 20 imagens e roda **Tesseract** e **EasyOCR** sobre o mesmo processamento, para
comparar o desempenho dos dois OCRs.

> As pastas `resultados/` e `placas_segmentadas/` contêm a saída da última execução e são
> regeneradas pelo código (ver "Como executar").

## Pipeline

```
imagem -> detecção da placa -> retificação -> ROI dos caracteres -> pré-processamento (7 variantes)
       -> OCR (Tesseract | EasyOCR) -> pós-processamento (LLLNLNN) -> votação -> texto final
```

| Etapa | Arquivo | O que faz |
|---|---|---|
| Detecção | `src/segmentation.py` | Limiarização HSV da faixa azul da placa, morfologia (close/open), `findContours`. Cada contorno recebe um score (proporção, bandeira verde/amarela, corpo claro, tamanho) e o melhor é escolhido. |
| Extração | `src/segmentation.py` | `minAreaRect` + `warpPerspective` endireitam a placa. A moldura tracejada inferior é achada por perfil de linhas escuras, assim o recorte não inclui a placa original do carro que aparece por baixo (imagens 5, 7 e 20). |
| ROI | `src/segmentation.py` | Recorta só os caracteres (remove faixa azul, moldura e o "BR"). |
| Pré-processamento | `src/preprocessing.py` | 7 variantes (abaixo). |
| OCR | `src/ocr_engines.py` | Mesma interface para os dois motores: `recognize(img) -> (texto, confiança)`. |
| Pós-processamento | `src/postprocess.py` | Limpa o texto, escolhe a melhor subsequência de 7 caracteres e corrige letra/dígito por posição (padrão Mercosul `LLLNLNN`). |
| Votação | `src/postprocess.py` | Voto por posição entre as variantes, peso = 0,1 + confiança do OCR. |

**Variantes de pré-processamento:** `gray`, `clahe`, `otsu`, `adaptive`, `bgnorm_otsu`,
`bgnorm_median`, `bgnorm_close`. A normalização de fundo (`bgnorm_*`) divide a imagem pelo fundo
estimado com *closing* morfológico, o que remove o degradê de iluminação. A mediana e o *closing*
suavizam a hachura da marca d'água dentro dos caracteres.

**Configuração do Tesseract:** `--oem 3 --psm 7`, whitelist `A-Z0-9`, dicionários desligados
(`load_system_dawg=0`, `load_freq_dawg=0`), altura do texto normalizada em 48 px.

## Estrutura do repositório

```
imagens/              20 imagens de entrada
src/                  segmentation, preprocessing, ocr_engines, postprocess, pipeline
gabarito.csv          placas reais (transcrição manual), usado só na avaliação
main.py               roda os dois pipelines + comparação
run_tesseract.py      pipeline completo com Tesseract
run_easyocr.py        pipeline completo com EasyOCR
compare_results.py    métricas contra o gabarito
requirements.txt
```

Gerado ao executar:

```
placas_segmentadas/   deteccao/, placas_extraidas/, rois_caracteres/
resultados/
  tesseract/resultados_tesseract.txt   texto final por imagem
  easyocr/resultados_easyocr.txt       texto final por imagem
  */resultados_*_detalhado.txt         saída bruta de cada variante
  comparacao.md, comparacao_variantes.png
```

## Como executar

Requisitos: Python 3.10+ e o [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) instalado.
No Windows, `C:\Program Files\Tesseract-OCR\tesseract.exe` é detectado automaticamente; em outro
caminho, defina `TESSERACT_CMD`.

```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

python main.py                 # os dois pipelines + comparação
```

Ou separadamente:

```bash
python run_tesseract.py        # só Tesseract  (~20 s)
python run_easyocr.py          # só EasyOCR    (baixa os modelos na 1ª vez)
python compare_results.py      # gera resultados/comparacao.md
```

A segmentação é refeita a cada execução, a partir das imagens originais.

## Resultados

Gabarito (`gabarito.csv`) transcrito manualmente. A segmentação localizou e extraiu **20/20 placas**.
Tabelas completas em `resultados/comparacao.md`.

### Pipeline final (7 variantes + votação + formato Mercosul)

| Motor | Placas exatas | Acurácia por caractere | CER | Tempo (20 imagens, CPU) |
|---|---|---|---|---|
| Tesseract | 18/20 (90%) | 98,6% | 1,4% | 43,1 s |
| EasyOCR | 16/20 (80%) | 97,1% | 2,9% | 64,1 s |

Erros do pipeline final:

| Imagem | Gabarito | Tesseract | EasyOCR |
|---|---|---|---|
| 3 | IYZ2Z18 | LYZ2Z18 | ok |
| 7 | GXQ2J95 | GXQ2I95 | GXO2J95 |
| 13 | ABP8Q89 | ok | ABP8O89 |
| 15 | THY7E19 | ok | THY2E19 |
| 20 | TGG1D44 | ok | TGG1O44 |

### Efeito do pré-processamento (placas exatas por variante, sem votação, após formato LLLNLNN)

| Variante | Tesseract | EasyOCR |
|---|---|---|
| gray | 7/20 | 8/20 |
| clahe | 4/20 | 9/20 |
| otsu | 12/20 | 11/20 |
| adaptive | 13/20 | 14/20 |
| bgnorm_otsu | 16/20 | 15/20 |
| bgnorm_median | 17/20 | 10/20 |
| bgnorm_close | 16/20 | 14/20 |

### Efeito do pós-processamento (mesmas saídas brutas)

| Motor | Só texto limpo | Formato, tabela genérica (G→6) | Formato, ajustado à fonte (G→0) |
|---|---|---|---|
| Tesseract | 10/20 | 14/20 | 18/20 |
| EasyOCR | 7/20 | 16/20 | 16/20 |

## Análise

- **Resultado geral:** o Tesseract acertou 18/20 e o EasyOCR 16/20. A diferença é de 2 placas em 20,
  pequena demais para concluir superioridade.
- **Tipo de erro:** o Tesseract errou `I`→`L` e `J`→`I` (letras de traço simples). O EasyOCR errou
  `Q`→`O` (2x), `D`→`O` e `7`→`2`, ou seja, confunde letras de contorno arredondado e um dígito.
  Como o `O` é letra, o formato LLLNLNN não corrige esses casos.
- **Sensibilidade ao pré-processamento:** o Tesseract depende bastante dele: vai de 4/20 (`clahe`) a
  17/20 (`bgnorm_median`). O EasyOCR é mais estável em imagens em tons de cinza (`clahe` 9/20 contra
  4/20 no Tesseract), mas piora com a mediana (10/20), que suaviza demais os traços. Para os dois, as
  variantes com normalização de fundo foram as melhores, o que confirma que o problema principal do
  dataset é a iluminação desigual.
- **Votação:** o resultado final superou a melhor variante isolada nos dois motores
  (Tesseract 18 contra 17; EasyOCR 16 contra 15).
- **Pós-processamento pelo formato:** foi o maior ganho para ambos (Tesseract 10→14/20 e EasyOCR
  7→16/20 só com a tabela genérica), pois corrige trocas letra/dígito por posição.
- **Zero com barra:** o Tesseract lê esse zero como `G`. O ajuste `G`→`0` levou o Tesseract de
  14/20 para 18/20 e não alterou o EasyOCR (16/20). **Com a tabela genérica, o EasyOCR (16/20)
  supera o Tesseract (14/20).** O ranking final depende, portanto, desse ajuste.
- **Tempo:** o EasyOCR levou cerca de 49% a mais (64,1 s contra 43,1 s) em CPU.

## Limitações

- A conversão `G`→`0` foi definida **depois de observar os erros nestas 20 imagens** e favorece o
  Tesseract (ver análise acima). O `compare_results.py` reporta as duas versões.
- Os parâmetros da segmentação (faixas de cor, margens da ROI) foram calibrados neste dataset.
  O detector depende da faixa azul e da bandeira, portanto não serve para placas cinza antigas.
- O gabarito é manual e pode conter erros de transcrição.
- Dataset de 20 imagens: diferenças de 1 ou 2 placas não são estatisticamente significativas.
- "Formato válido 20/20" no `comparacao.md` só indica que o texto segue LLLNLNN (o pós-processamento
  força isso), não que a placa está correta.