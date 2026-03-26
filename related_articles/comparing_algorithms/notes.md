### Resumo

O estudo avalia o desempenho de três algoritmos de *Machine Learning* — Random Forest (RF), k-Nearest Neighbors (kNN) e Extreme Gradient Boosting (XGBoost) — para classificar explosões solares nas categorias B, C, M e X. Este artigo utiliza 13 parâmetros magnéticos provenientes do dataset SHARP (SDO/HMI).

A pesquisa inova ao combinar tarefas de classificação binária (B/C vs. M/X) e multiclasse. Além disso, os autores utilizam *Principal Component Analysis* (PCA) para redução de dimensionalidade, comparando o impacto de usar 8 componentes (95% da variância) versus 100 componentes (97.5% da variância). A conclusão aponta que XGBoost e RF apresentam os melhores desempenhos, especialmente quando a dimensionalidade dos dados é maior (100 componentes). O XGBoost destacou-se por ter os resultados mais consistentes na detecção das explosões de alto impacto (M e X).

### Validação e Métricas Utilizadas

**1. Metodologia de Validação:**

**Tratamento de Desbalanceamento:** Para contornar a diferença drástica no número de amostras, os autores não utilizaram pesos diretamente no modelo para esse fim primário. Eles criaram 100 *datasets* diferentes, onde realizaram uma amostragem aleatória repetida das classes majoritárias para forçar o balanceamento com as classes minoritárias, tanto na avaliação binária quanto multiclasse.

**Validação Cruzada:** Foi aplicada uma Validação Cruzada Estratificada de 10 dobras (*10-fold Stratified Cross-Validation*). Nove dobras foram usadas para treino e uma para validação iterativamente.

**Otimização:** Utilizaram *GridSearch* iterado juntamente com as 10 dobras para encontrar os melhores hiperparâmetros para cada modelo.



**2. Métricas Extraídas:**
A média das seguintes métricas nas 10 dobras foi usada para avaliar o desempenho final:

**Acurácia Global (Overall Accuracy):** Mede o acerto geral.

**F1 Score:** Utilizado extensivamente no artigo (incluindo F1 ponderado para multiclasses), sendo uma excelente métrica para lidar com datasets de explosões solares devido à sua capacidade de equilibrar precisão e *recall*.

**PR AUC (Precision-Recall Area Under the Curve):** Fundamental para dados altamente desbalanceados. Mede a curva entre a precisão e a taxa de verdadeiros positivos.

**ROC AUC (Receiver Operating Characteristic Area Under Curve):** Mede a capacidade do modelo de discriminar entre as classes.

*Precisão e Sensibilidade (Recall)* também foram calculadas de forma subjacente para compor o PR AUC e o F1 Score.



### Relevância e Impacto para o meu Projeto

**Score: 9.5 / 10**

**Justificativas:**

**Sinergia de Algoritmo:** O XGBoost é o modelo base do meu projeto e também foi considerado o modelo mais robusto neste artigo. é possível comparar diretamente os *scores* alcançados por eles.

**Mesma Divisão do Problema:** Assim como a arquitetura ("Solarfall") separa as tarefas entre classificações binárias (Gatekeeper) e distinção de classes maiores , este estudo foca em comparar uma visão binária (Baixo Impacto B/C vs. Alto Impacto M/X) contra uma visão multiclasse total.

**Base para o tratamento de mags:** O artigo analisado foca em atributos magnéticos (dados vetoriais SDO/HMI e matrizes de força de Lorentz). Pode servir como referência sobre como tratar e avaliar as features magnéticas.

### Relevância na Comunidade Científica

**Artigo muito recente e autores não tão bem estabelecidos**