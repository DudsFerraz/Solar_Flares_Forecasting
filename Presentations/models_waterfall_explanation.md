# Sumarização da Arquitetura de Previsão

As previsões serão realizadas através de uma cascata de 3 modelos de machine learning:

### Modelo 1 - 'Gatekeeper' (O Porteiro)

* **Objetivo:** Separar 'No Flare' (calmaria total) de 'Flare' (qualquer nível de atividade).
* **Lógica:** É treinado em um problema binário (Classe 0 vs. Classes 1-5). O objetivo principal é filtrar os ~26% dos dados que são "calmaria total" e passar os outros ~74% (que contêm "alertas reais" e "alarmes falsos") adiante.
* **Métrica de Otimização:** Foco total em maximizar o Recall da classe 'Flare', aceitando um alto número de Falsos Positivos.

### Modelo 2 - 'Great Filter' (O Filtro)

* **Objetivo:** Separar explosões de baixo impacto ('A', 'B', 'C') de explosões de alto impacto ('M', 'X').
* **Lógica:** É treinado apenas com as linhas classificadas como 'Flare'. Sua função é filtrar o "ruído" (explosões ABC e os Falsos Positivos do Modelo 1).
* **Balanceamento:** Este modelo lida com dados muito mais balanceados que os outros.

### Modelo 3 - 'Specialist' (O Especialista)

* **Objetivo:** Separar 'M' de 'X'.
* **Lógica:** É treinado apenas com as linhas classificadas como 'MX'.
* **Balanceamento:** Estes dados voltam a ser relevantemente desbalanceados (mais M do que X), exigindo foco especial na detecção da classe 'X', que é a mais rara e perigosa.

As explosões classificadas como A, B, C serão descartadas por serem consideradas de baixíssimo impacto aos seres humanos, mesmo em satélites ou missões espaciais. A cascata é robusta o suficiente para que, mesmo que se altere os dados de entrada (engenharia de features), possamos seguir com a arquitetura dos 3 modelos.

---

## Justificativa da Escolha do Algoritmo: XGBoost

Para a implementação inicial dos três modelos, optamos pelo XGBoost (eXtreme Gradient Boosting), um algoritmo de gradient boosting que se enquadra na categoria de métodos de ensemble learning.

A escolha é baseada em cinco vantagens principais para o nosso problema específico:

### Vantagem 1: Desempenho Bruto em Dados Tabulares (O Padrão-Ouro)

Por uma década, o XGBoost (e seus concorrentes de boosting, como o LightGBM) tem sido o vencedor absoluto em competições de dados tabulares (como as do Kaggle). Ele é excepcional em encontrar padrões complexos e não-lineares.

### Vantagem 2 (e principal): Gerenciamento Nativo de Desbalanceamento

Nossa cascata é uma série de problemas de desbalanceamento. O XGBoost foi construído para isso. O parâmetro `scale_pos_weight` (usado nos Modelos 1 e 3) é a ferramenta de força bruta mais eficaz para este cenário. Alternativas como RandomForest têm `class_weight`, mas o boosting (focar nos erros) do XGBoost é inerentemente mais poderoso para aprender as classes raras.

### Vantagem 3: Interpretabilidade

O gráfico de `feature_importance` permite explicar parcialmente o que leva a uma previsão. Isso é crucial para um trabalho científico. Outros modelos (como SVMs ou Redes Neurais) são "caixas-pretas", tornando o relatório científico muito mais difícil de escrever.

### Vantagem 4: Escalabilidade e Velocidade (GPU)

XGBoost é escrito em C++, é multithreaded (`n_jobs=-1`), e o mais importante: suporta o uso de GPU. Treinar o RandomForest do scikit-learn em milhões de linhas seria dolorosamente lento (só CPU). O XGBoost pode fazer isso em minutos.

### Vantagem 5: Robustez

Sendo o XGBoost um modelo baseado em árvores, ele não requer normalização ou padronização de features (como `MinMaxScaler`), mesmo com features em escalas totalmente diferentes (`xl_mean_30D` é muito diferente de `xl_deriv_5min`). Modelos como Regressão Logística e SVMs falhariam catastroficamente sem isso. O XGBoost "apenas funciona".

---

## Apêndice A: Estratégias de Otimização Futuras

A escolha do XGBoost para os três modelos é a nossa linha de base (baseline). Otimizações futuras se concentrarão em duas áreas: substituir o Modelo 2 por uma alternativa mais leve e testar o LightGBM como um concorrente direto do XGBoost.

### A Oportunidade do Modelo 2 ("Great Filter")

O Modelo 2 (ABC vs. XM) é o candidato ideal para substituição.

* **Por quê?** Ele resolve o problema mais "fácil" da cascata. Os dados que ele recebe já foram filtrados pelo "Porteiro" (Modelo 1). O desbalanceamento entre as classes ABC (aprox. 56%) e XM (aprox. 18%) é significativamente menos severo do que nos outros dois modelos.
* **O Risco:** Usar um algoritmo pesado como o XGBoost aqui pode ser um "overkill" (exagero), gastando tempo computacional desnecessário.
* **Alternativas Viáveis:**
    * **RandomForest:** Seria mais rápido de otimizar (menos hiperparâmetros) e ainda oferece boa performance e interpretabilidade.
    * **Regressão Logística:** Seria o modelo mais rápido e leve. Como o problema é menos complexo, um modelo linear (combinado com nossas features não-lineares) pode ser "bom o suficiente" e nos daria interpretabilidade total através dos coeficientes das features.

### O Concorrente Direto: LightGBM (LGBM)

O LightGBM é o principal concorrente do XGBoost e deve ser testado em todos os três estágios da cascata.

* **Por quê?** Velocidade. O LGBM é famoso por ser significativamente mais rápido que o XGBoost em datasets grandes como o nosso.
* **A Diferença Técnica:**
    * **XGBoost (Level-wise):** Constrói árvores de forma simétrica, nível por nível. É mais robusto, mas mais lento.
    * **LightGBM (Leaf-wise):** Constrói árvores de forma assimétrica. Ele foca em dividir as "folhas" (ramos) que têm o maior erro, ignorando ramos que já são "bons".
* **O Trade-off:** A estratégia leaf-wise do LGBM é mais rápida, mas mais propensa a overfitting em datasets pequenos. Como nosso dataset é grande, esse risco é reduzido, e podemos nos beneficiar de sua velocidade sem, provavelmente, sacrificar a performance.
* **Plano de Teste:** Após estabelecer o baseline com XGBoost, devemos executar o mesmo tuning do Optuna para o `lgb.LGBMClassifier` e comparar o tempo de treino e as métricas de avaliação (Recall, Precision, F1).