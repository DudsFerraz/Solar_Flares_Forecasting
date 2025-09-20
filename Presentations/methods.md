# Simple Exponential Smoothing

## **_Pontos negativos:_**
* #### Extremamente ineficaz

---

## ~~**_Pontos positivos:_**~~

---

# Double Exponential Smoothing

## **_Pontos negativos:_**
* #### Extremamente ineficaz

---

## ~~**_Pontos positivos:_**~~

---

# Holt Winters (Triple Exponential Smoothing)

## **_Pontos negativos:_**
* #### Não suporta múltiplas sazonalidades.
* #### Não suporta regressão para refinamento do modelo.
* #### Baseia-se 100% nos dados históricos da série temporal, sem considerar variáveis externas.

---

## **_Pontos positivos:_**
* #### Lida bem com tendência e sazonalidade.
* #### Implementação simples.
* #### Adaptável através de parâmetros.
* #### Eficiente para problemas simples e bem definidos.

## **_Parâmetros:_**

---

# SARIMAX

## **_Pontos negativos:_**
* #### Ruim para séries temporais longas.
* #### Não suporta múltiplas sazonalidades.

---

## **_Pontos positivos:_**
* #### Implementação fãcil.
* #### Precisão decente.

---

## **_Parâmetros:_**

---

# Prophet

## **_Pontos negativos:_**
* #### Necessita de otimização.
* #### Ruim para dinâmicas de curto-prazo.

---

## **_Pontos positivos:_**
* #### Flexível.
* #### Built-in Cross Validation.
* #### Permite adição de eventos dinâmicos (ex: feriados).
* #### Permite adição de regressores.

---

## **_Parâmetros:_**
Parâmetros de Sazonalidade e Feriados
yearly_seasonality = True, weekly_seasonality = True

O que é? Instruções para o Prophet incluir automaticamente componentes para capturar padrões que se repetem anualmente e semanalmente. Por baixo dos panos, ele faz isso criando Termos de Fourier.

Impacto no Desempenho: Essencial para capturar os ciclos básicos do seu negócio. yearly_seasonality irá modelar a diferença entre verão e inverno, enquanto weekly_seasonality modelará a diferença entre dias de semana e fins de semana.

O que mudar? Você pode desligar um deles (False) se não acreditar que o padrão existe. Também é possível especificar a "flexibilidade" da sazonalidade. Por exemplo, yearly_seasonality=20 usaria mais termos de Fourier, permitindo um padrão anual mais complexo.

holidays = df_holidays

O que é? Permite que você passe um DataFrame contendo as datas de feriados ou eventos especiais que afetam a série.

Impacto no Desempenho: Extremamente útil para modelar picos ou quedas bruscas em dias específicos que não são capturados pela sazonalidade regular. O Prophet aprenderá um efeito específico para cada feriado.

O que mudar? Você pode adicionar mais eventos ao seu df_holidays ou ajustar o lower_window e upper_window para capturar o impacto nos dias ao redor do feriado (ex: o efeito do Natal começa dias antes).

seasonality_mode = 'multiplicative'

O que é? Define como os componentes sazonais são combinados com a tendência. O padrão é 'additive' (tendência + sazonalidade). O modo 'multiplicative' (tendência * (1 + sazonalidade)) é usado quando a amplitude dos picos sazonais cresce junto com a tendência.

Impacto no Desempenho: Para dados de aluguel de bicicletas, é muito provável que o pico do verão seja muito maior (em números absolutos) em um ano de alta demanda do que em um ano de baixa demanda. Portanto, 'multiplicative' é uma excelente escolha aqui.

O que mudar? Se as flutuações sazonais tivessem uma altura mais ou menos constante ao longo dos anos, você voltaria para o modo 'additive'.

Parâmetros de Ajuste (Prior Scales)
Esses são parâmetros de regularização. Eles controlam a "flexibilidade" dos componentes, ajudando a evitar o superajuste (overfitting).

seasonality_prior_scale = 10, holidays_prior_scale = 10

O que é? Controlam a flexibilidade dos componentes de sazonalidade e feriados.

Impacto no Desempenho: O valor padrão é 10. Um valor maior (ex: 20) permite que os efeitos de sazonalidade/feriados se ajustem mais fortemente aos dados, capturando picos maiores (risco de overfitting). Um valor menor (ex: 1.0) "amortece" os efeitos, tornando-os mais suaves (risco de underfitting).

O que mudar? Se você plotar os componentes do modelo (m.plot_components(forecast)) e achar que a sazonalidade está muito "tímida" e não captura a magnitude real dos picos, aumente esse valor. Se parecer muito ruidosa, diminua.

changepoint_prior_scale = 0.05

O que é? Este é um dos parâmetros mais importantes. Ele controla a flexibilidade da tendência. O Prophet detecta automaticamente pontos onde a taxa de crescimento da sua série muda (changepoints).

Impacto no Desempenho: O padrão é 0.05. Um valor maior (ex: 0.5) tornará a linha de tendência muito mais flexível, permitindo que ela mude de direção mais drasticamente. Um valor menor (ex: 0.01) a tornará mais "rígida".

O que mudar? Se seu modelo parece não capturar mudanças recentes na tendência dos dados (sub-ajuste), aumente este valor. Se a tendência parece estar se ajustando demais ao ruído (super-ajuste), diminua este valor.

Regressores Adicionais
m.add_regressor(...)

O que é? Permite adicionar outras colunas (variáveis externas) que você acredita que ajudam a prever sua variável y. O Prophet as adiciona como um componente linear ao modelo.
Impacto no Desempenho: Essencial para capturar efeitos que não são puramente temporais, como o clima. Se a temperatura sobe, a demanda por bicicletas aumenta, independentemente de ser terça-feira ou sábado.
---

# SVM/SVC
Uma fonte comum de confusão para iniciantes é a distinção entre os termos SVM, SVC e SVR. É crucial entender que "Máquina de Vetores de Suporte" (SVM) refere-se à teoria e metodologia geral subjacente. O SVM é o framework conceitual baseado nos princípios de encontrar um hiperplano de margem máxima em um espaço de características de alta dimensão.   

Por outro lado, "Classificador de Vetores de Suporte" (Support Vector Classifier - SVC) e "Regressão de Vetores de Suporte" (Support Vector Regression - SVR) são as aplicações ou implementações específicas dessa teoria para resolver problemas distintos.   

SVC é a aplicação do SVM para tarefas de classificação, onde o objetivo é prever um rótulo de classe discreto (por exemplo, "spam" ou "não spam").

SVR é a aplicação do SVM para tarefas de regressão, onde o objetivo é prever um valor numérico contínuo (por exemplo, o preço de uma ação).

Em resumo, SVM é o nome da família de algoritmos, enquanto SVC e SVR são membros específicos dessa família, cada um adaptado para um tipo de tarefa de aprendizado supervisionado.


## **_Pontos negativos:_**
* #### Necessário a aplicação de técnicas como "embedding" da série temporal ou criação de uma janela deslizante, que convertem efetivamente o problema de previsão num problema de regressão padrão.
* #### Custo Computacional: dependendo da implementação, o treinamento pode ser proibitivamente lento para conjuntos de dados muito grandes.

---

## **_Pontos positivos:_**
* #### Boa Capacidade de Generalização.
* #### Robustez a Outliers.
* #### Captura de não linearidade. 

---

## **_Parâmetros:_**

---

## **_Notas:_**
* Baseando-se na biblioteca python "sklearn.svm" utilizar LinearSVC() no lugar de SVC(). Enquanto um tem complexidade algoritmica quadrática, o outro é linear. O desempenho parece similar.
---
* A literatura acadêmica fornece evidências mistas, mas geralmente positivas, sobre o uso do SVR para previsão de séries temporais, destacando tanto seu potencial quanto a importância de uma aplicação cuidadosa. Estudos no domínio financeiro, conhecido por suas séries temporais não-lineares e ruidosas, frequentemente demonstram que o SVR é uma ferramenta bem-sucedida para previsão. A capacidade do SVR de lidar com a não-estacionariedade e a multi-escala dos dados financeiros é uma vantagem chave.
---
* A literatura aponta para uma tendência clara: o SVR atinge o seu potencial máximo quando integrado em **modelos híbridos**.
---

# Mensurando Erros e Acertos

---

## Accuracy

---
* #### Métrica de Classificação.
* #### A porcentagem de previsões que o modelo acertou do total de amostras.
* #### Classificações corretas / Todas as classificações
* #### Pode ser enganosa em datasets desbalanceados.
* #### Accuracy = (TP + TN) / Total of samples.


---

## Precision

---
* #### Métrica de Classificação.
* #### Das vezes que o modelo previu a classe positiva, quantas ele acertou? Mede a confiabilidade das previsões positivas.
* #### Essencial quando o custo de um Falso Positivo é alto.
* #### Precision = TP / (TP + FP)

---

## Recall

---
* #### Métrica de Classificação.
* #### De todas as instâncias que eram realmente positivas, quantas o modelo conseguiu identificar? Mede a capacidade do modelo de "encontrar" a classe de interesse.
* #### Essencial quando o custo de um Falso Negativo é alto
* #### Recall = TP / (TP + FN)

---

## F1 Score

---
* #### Métrica de Classificação.
* #### A média harmônica entre Precisão e Recall, fornecendo um único score que equilibra ambas as métricas.
* #### É a métrica de escolha para muitos problemas com classes desbalanceadas, pois penaliza modelos extremos numa métrica e fracos na outra.
* #### F1 Score = 2 * ( (Precision * Recall) / (Precision + Recall) )

---

## AUC (Area Under the ROC Curve)

---
* #### Métrica de Classificação.
* #### A Área Sob a Curva ROC. Mede a capacidade do modelo de distinguir entre as classes positiva e negativa em todos os limiares de classificação.
* #### Ótima para comparar o poder discriminatório geral de diferentes modelos, pois é independente do limiar de classificação escolhido.
* #### Fórmula: Integral da curva ROC (plot da taxa de VP vs. taxa de FP). Um valor de 0.5 representa um modelo aleatório; 1.0 representa um modelo perfeito.

---

## Cross-Entropy / Log Loss

---
* #### Métrica de Classificação.
* #### Mede a performance de um modelo que retorna probabilidades. Penaliza fortemente previsões erradas feitas com alta confiança.
* #### É uma função de perda, portanto, quanto menor, melhor.
* #### Fórmula: _Complicada demais para colocar aqui._

---

## Mean Absolute Error (MAE)

---
* #### Métrica de Regressão.
* #### A média das diferenças absolutas entre os valores reais e os previstos.
* #### Fácil de interpretar, pois está na mesma unidade da variável alvo. É menos sensível a outliers que o RMSE.

---

## Root Mean Squared Error (RMSE)

---
* #### Métrica de Regressão.
* #### A raiz quadrada da média dos erros ao quadrado.
* #### A métrica de regressão mais comum. Penaliza erros grandes mais severamente que o MAE devido à exponenciação ao quadrado.

---

## Mean Absolute Percentage Error (MAPE)

---
* #### Métrica de Regressão.
* #### A média dos erros percentuais absolutos.
* #### Expressa o erro em porcentagem, o que é muito intuitivo para stakeholders. Torna-se instável ou indefinido se os valores reais (yi) forem zero ou muito próximos de zero.

---

## R² (R-Quadrado / Coeficiente de Determinação)

---
* #### Métrica de Regressão.
* #### A proporção da variância na variável alvo explicada pelo modelo.
* #### Varia de -∞ a 1. Um valor de 1 indica que o modelo explica toda a variabilidade. Um valor de 0 indica que o modelo não é melhor que simplesmente prever a média.

---

## Similaridade de Cossenos (Cosine Similarity)

---
* #### Métrica de Regressão.
* #### Mede o cosseno do ângulo entre dois vetores não nulos. Usado para medir a similaridade de orientação, independentemente da magnitude.
* #### Varia de −1 (direções opostas) a 1 (mesma direção), com 0 indicando ortogonalidade. Amplamente utilizado em NLP para comparar a similaridade semântica de documentos.

---