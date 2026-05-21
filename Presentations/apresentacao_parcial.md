# Investigando os aspectos mais relevantes para a previsão de explosões solares

## 👨‍💻 Sobre o Pesquisador

> ### **Eduardo Ferraz de Campos**
> 
> 
> **20 anos | Piracicaba - SP**
> * **Formação:** Graduando em Engenharia de Software (5º Período) – **IFSP**.
> * **Acadêmico:** Pesquisador no **IFSP** sob orientação do **Dr. Sérgio Luisir Díscola Junior**.
> * **Atuação Profissional:** Estagiário na **USP São Carlos**.
> 
> 
> **Nota de Escopo:** Minha abordagem é fundamentada na **Engenharia de Software e Machine Learning**. O foco reside no processamento de dados e na eficácia dos modelos, tratando fenômenos astrofísicos como variáveis de entrada em sistemas complexos.

## 🛰️ Abordagens de Previsão

Na literatura de clima espacial, existem duas frentes principais para prever o comportamento do Sol:

| Perspectiva                                    | Lógica           | Dados Utilizados                                                        |
|------------------------------------------------|------------------|-------------------------------------------------------------------------|
| **Causa $\rightarrow$ Efeito**                 | Física Magnética | **Magnetogramas** para prever raios-X futuros.                          |
| **Efeito Passado $\rightarrow$ Efeito Futuro** | Séries Temporais | **Fluxo de Raios-X** passados para prever **Fluxo de Raios-X** futuros. |

Minha pesquisa investiga a eficácia de ambas as abordagens, isoladas e combinadas.

---

## 📊 Natureza dos Dados

### A. Fluxo de Raios-X (Dados Globais)

Dados do satélite **GOES (NOAA)**, especificamente o canal **XRS-B** (0.1 a 0.8 nm).

* **Comportamento:** É um índice **global**.
* **Limitação:** O sensor captura a irradiância total; não sabemos *onde* no Sol a energia nasceu, apenas sua intensidade final.

### B. Magnetogramas SHARP (Dados Regionais)

Dataset `hmi.sharp_720s` (Spaceweather HMI Active Region Patch).

* **O diferencial:** Diferente do raio-X, o SHARP identifica **Regiões Ativas (ARs)** específicas.
* **Complexidade:** Em um mesmo timestamp, podemos ter várias linhas de dados, cada uma representando uma região ativa diferente presente na face solar.

---

## 🕵️‍♂️🔍 Estratégia de Experimentação

Construímos diferentes arquiteturas para testar o que realmente importa para o modelo:

| Família de Modelo              | Insumo (Input)                            | Alvo (Target)   | Objetivo                                        |
|--------------------------------|-------------------------------------------|-----------------|-------------------------------------------------|
| **X-Ray**                      | Fluxo Global                              | Global          | Validar a força da série temporal.              |
| **Mag Global (Max)**           | Maior valor magnético do disco            | Global          | A região mais forte define o Sol todo?          |
| **Mag Global (Sum)**           | Soma de todas as regiões (fluxo global)   | Global          | O acúmulo de energia total é relevante?         |
| **Mag Regional + MGS + X-Ray** | Dados de uma região específica            | Regional/Global | Previsão global baseado em previsões regionais. |
| **Modelo Final**               | A melhor combinação de features e modelos | Regional/Global | O "Estado da Arte" da pesquisa.                 |

---

# 🏗️ Arquitetura Solarfall

A **Solarfall** é uma arquitetura original baseada numa **estratégia de cascata (hierárquica)**. Em vez de um único modelo tentar resolver toda a complexidade do Sol, dividimos o problema em 4 especialistas que filtram os dados progressivamente.

## 🌊 A Cascata

| Ordem  | Modelo             | Missão Principal                                        | Foco Estratégico                                                                      |
|--------|--------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------|
| **1º** | **Gatekeeper**     | Separar "Calmaria" de "Alerta".                         | **Recall alto:** Não deixar nenhuma explosão passar, mesmo que gere falsos positivos. |
| **2º** | **Great Filter**   | Separar baixo impacto (A, B) de alto impacto (C, M, X). | **Limpeza de Ruído:** Eliminar explosões irrelevantes e erros do Gatekeeper.          |
| **3º** | **Specialist 910** | Separar Classe C de Classes M/X.                        | **Refinamento:** Identificar o limiar de perigo real.                                 |
| **4º** | **Specialist MX**  | Separar Classe M de Classe X.                           | **Precisão Extrema:** Diferenciar eventos fortes de eventos catastróficos.            |

> **Vantagem:** Cada modelo é treinado com um dataset mais balanceado e específico, permitindo que a inteligência foque apenas nos padrões que diferenciam aquelas classes específicas.

---

## ⚙️ Metodologia de Treinamento

**1. Particionamento Cronológico**

* Respeito rigoroso à linha do tempo: **70% Treino / 15% Validação / 15% Teste**.
* Evita o *data leakage* (vazamento de dados), garantindo que o modelo não "preveja o passado".

**2. Seleção Automática de Atributos (Quick Scan)**

* Uso de **XGBoost** para identificar a importância das features.
* Mantemos apenas o subconjunto que representa **95% da importância cumulativa**.
* **Resultado:** Redução de ruído estatístico e maior eficiência computacional.

**3. Otimização e Calibração**

* **Optuna:** Busca Bayesiana para encontrar os melhores hiperparâmetros de forma inteligente, não aleatória.
* **Ajuste de Limiar (Threshold):** O padrão $0.5$ é descartado. Cada modelo tem seu "gatilho" ajustado matematicamente para equilibrar métricas conforme sua função na cascata.

---

# 📈 Métricas de Classificação

## 📋 Análise de Relatório de Classificação

**Precision (Precisão)**
* **Essência:** Quando o modelo diz "Sim", qual a chance de ele estar certo? Foca em evitar alarmes falsos (Falsos Positivos).
* **Exemplo:** O modelo indicou 'Flare' 10 vezes. Ao checar, em 8 vezes havia 'Flare'', e nas outras 2 eram alarmes falsos. **Precisão de 80%**.

**Recall (Revocação)**
* **Essência:** Das explosões que aconteceram, quantas o modelo conseguiu capturar? Foca em não deixar nada passar (Falsos Negativos).
* **Exemplo:** Ocorreram 10 explosões. O modelo detectou 7, mas ficou mudo nos outros 3 (omissão). **Recall de 70%**.

**Trade-off:**
Se o modelo for supersensível, para não perder nada (alto Recall), ele vai tocar muitos alarmes falsos (baixa Precisão). Se você o deixa extremamente rigoroso para nunca ter alarme falso (alta Precisão), ele pode ignorar uma explosão relevante (baixo Recall). A escolha depende do que custa mais caro para o problema: o alarme falso ou a omissão.

**F1 Score**

* **Essência:** É o meio-termo (média harmônica) entre a Precisão e o Recall. Resume o "trade-off" em uma nota única.
* **Exemplo:** Se o modelo for configurado para ter 90% de Precisão e apenas 10% de Recall, o F1 Score cairá drasticamente, evidenciando que a balança está desequilibrada.

---

## ❌ Análise de Erros

**Zoneamento de Fluxo (`analyze_flux_errors`)**

* **Essência:** Identifica *onde* o modelo mais erra em relação à intensidade do evento, dividindo os dados em zonas (segura, transição, perigo).
* **Exemplo:** O modelo não teve erros na zona de 'Perigo', mas gerou 6.000 falsos alarmes na 'Zona Segura'. Ou seja, ele se assusta à toa com fluxos de energia muito baixos.

**Distribuição por Classe Solar (`analyze_error_distribution`)**

* **Essência:** Mede a gravidade das omissões do modelo com base na classificação oficial (C, M, X), revelando se os erros ocorrem nos piores cenários.
* **Exemplo:** O modelo deixou passar mais de 1.000 explosões da classe X (a mais destrutiva). Isso evidencia um ponto cego crítico para eventos extremos.

---

## 📶 Métricas Avançadas

**TSS (True Skill Statistic)**

* **Essência:** Mede a capacidade de acertar os eventos reais sem disparar alarmes falsos. É a métrica ideal quando há um desbalanceamento gigante (muitos dias calmos, poucos dias de explosão).
* **Exemplo:** Um TSS de 0.30 significa que a taxa de acertos legítimos do modelo supera a taxa de alarmes falsos em 30% (onde 0 seria puro chute).

**HSS (Heidke Skill Score)**

* **Essência:** Compara a precisão do modelo com a precisão de um modelo "cego" que apenas chuta aleatoriamente as respostas.
* **Exemplo:** Um HSS de 0.30 indica que o modelo é 30% melhor do que jogar uma moeda para decidir se o alarme deve tocar ou não.

**MCC (Matthews Correlation Coefficient)**

* **Essência:** O "padrão-ouro" para avaliar dados desbalanceados. Ele resume a qualidade global do modelo em um único número, exigindo que o modelo seja bom em acertar tanto as explosões quanto a calmaria.
* **Exemplo:** Varia de -1 a 1. Um MCC de 0.30 mostra que existe uma inteligência real nas previsões, mas ainda há bastante margem para evolução (valores acima de 0.60 são considerados excelentes).

**ROC AUC e PR AUC**

* **Essência:** Avaliam a capacidade do modelo de separar o que é explosão do que não é, independentemente de onde você coloca o "limite" para disparar o alarme. A PR AUC foca quase exclusivamente na capacidade de achar as explosões.
* **Exemplo:** Uma PR AUC de 0.82 significa que, ao listar as previsões da mais provável para a menos provável, o modelo tem 82% de chance de colocar uma explosão real no topo da lista.
---

## 🔄 Baseline e Transições

**1. PR-F1 (Persistence Relative F1)**

* **Essência:** Coloca o modelo à prova de "preguiça" (persistência), que assume que o amanhã será exatamente igual ao hoje.
* **Exemplo:** Um PR-F1 negativo (-0.02) revela que o modelo complexo perdeu para a regra ingênua de "se teve explosão ontem, digo que terá hoje".

**2. AC (Activity Change) vs NC (No Change)**

* **Essência:** Isola o desempenho do modelo em momentos de inércia (o estado do Sol continuou igual) e momentos de virada (o Sol "ligou" ou "desligou" abruptamente).
* **Exemplo:** No estado estável (NC), o modelo pontua bem (MCC 0.48). Já na mudança abrupta (AC), ele erra mais do que acerta (MCC -0.11). Isso comprova que o modelo é ótimo em diagnosticar a inércia, mas sofre para antecipar uma virada de cenário.

