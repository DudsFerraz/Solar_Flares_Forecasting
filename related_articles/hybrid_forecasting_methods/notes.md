### Resumo

Este artigo consiste em uma **revisão sistemática de literatura** conduzida sob a metodologia PRISMA, investigando o estado da arte de métodos híbridos para previsão de séries temporais entre 2019 e 2022. O objetivo central dos autores foi verificar se a combinação de métodos estatísticos tradicionais (como **ARIMA/SARIMA**) com modelos de aprendizado profundo (como **LSTM/RNN/CNN**) de fato supera a performance de modelos isolados.

A revisão selecionou 21 trabalhos relevantes que aplicaram essas abordagens em diversos domínios, desde economia (PIB, taxas de câmbio) até ciências ambientais e saúde. O artigo destaca que a principal vantagem da hibridização é a capacidade de decompor a série temporal: modelos estatísticos lidam com os **componentes lineares**, enquanto as redes neurais capturam as **nuances não lineares** e resíduos complexos.

### Metrificação dos Resultados

Diferente de artigos que propõem um único modelo, este trabalho sintetiza as métricas mais aceitas pela comunidade científica para validar previsões quantitativas. Os autores focaram em duas métricas principais de erro para realizar a comparação direta entre os 21 estudos:

* **RMSE (Root Mean Squared Error)**: Utilizado para medir a magnitude do erro. Por penalizar erros maiores (devido ao quadrado da diferença), é ideal para entender a precisão absoluta do modelo em relação aos valores reais.


* **MAPE (Mean Absolute Percentage Error)**: Utilizado para medir a precisão em termos percentuais. É uma métrica de erro relativo que facilita a compreensão do impacto do erro independentemente da escala dos dados.



A conclusão principal do artigo, baseada nessas métricas, é que em **100% dos casos analisados**, o modelo híbrido obteve valores de RMSE e MAPE inferiores aos seus respectivos modelos base (individuais).

### Relevância e Impacto para o meu projeto

**Score: 5 / 10**

**Validação da Abordagem Híbrida**: Embora meu foco atual seja o XGBoost (um modelo de *ensemble*), o artigo reforça que a tendência científica atual é a **combinação de forças**. Isso me dá embasamento teórico para justificar por que estou utilizando uma arquitetura em cascata (*Solarfall*) em vez de um modelo simples de prateleira.

**Conexão Direta**: Citei este trabalho no meu relatório parcial para fundamentar que "não existe solução universal para previsões" e que a abordagem deve ser específica variar conforme a natureza dos dados. Justificando a criação da "Solarfall".

### Relevância na Comunidade Científica

****