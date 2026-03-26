### Resumo

O artigo realiza um abrangente *benchmark* comparando modelos baseados em árvores (como XGBoost e Random Forest) com novas e tradicionais arquiteturas de Deep Learning (como ResNet, FT_Transformer e SAINT) aplicadas a dados tabulares. Os autores padronizaram 45 conjuntos de dados de diversos domínios e conduziram uma busca massiva de hiperparâmetros. O estudo conclui que os algoritmos baseados em árvores continuam sendo o estado da arte para *datasets* tabulares de tamanho médio (aproximadamente 10.000 amostras), superando as redes neurais mesmo sem levar em conta sua enorme vantagem no tempo de treinamento.

O trabalho investiga empiricamente os motivos dessa disparidade, descobrindo que as redes neurais sofrem de um forte viés para soluções suaves, tendo dificuldade para aprender padrões e funções irregulares. Além disso, os modelos de Deep Learning perdem muita performance ao lidar com variáveis não informativas e têm sua eficiência prejudicada por serem invariantes à rotação dos dados, uma propriedade matemática que acaba ocultando as características intrínsecas individuais de *features* tabulares. Para estimular pesquisas futuras, os autores também disponibilizaram todos os resultados brutos da busca de hiperparâmetros, equivalente a 20.000 horas de processamento computacional.

### Metrificação dos Resultados

Para metrificar e comparar o desempenho de tantos algoritmos de forma isenta, mitigando a alta variância estatística comumente introduzida pela otimização de hiperparâmetros, a metodologia adotou as seguintes estratégias:

**Métricas de Desempenho Base**: Foi utilizada a Acurácia (*test set accuracy*) para as tarefas de classificação e o *score* R² para as tarefas de regressão.

**Métrica de Agregação (ADTM)**: Como os testes englobaram 45 *datasets* com Bayes rates e dificuldades bastante diferentes, os resultados precisaram ser normalizados entre 0 e 1 através de uma transformação afim. Para evitar distorções graves geradas por modelos com desempenho catastrófico, a normalização não ancorou o valor "0" no pior modelo de todos. Em vez disso, a escala foi construída entre o modelo de ponta e o modelo posicionado no quantil de 10% (para classificação) ou 50% (para regressão).

**Validação via Orçamento de Random Search**: A validação não comparou apenas os resultados finais isolados. Foram executadas cerca de 400 iterações de busca aleatória de hiperparâmetros. O desempenho foi plotado de acordo com o número $n$ de iterações executadas.

**Estimativa Robusta (Bootstrap)**: Para consolidar a estabilidade da métrica, o resultado ótimo alcançado no conjunto de validação foi avaliado no conjunto de teste 15 vezes distintas, reembaralhando a ordem da busca aleatória a cada vez. Isso gerou faixas consistentes com estimativas das pontuações máximas e mínimas esperadas daquele algoritmo.


### Relevância e Impacto para o meu projeto

**Score: 8.5 / 10**

**Justificativa Teórica Fundamental**: Este estudo é a espinha dorsal que alicerça a minha escolha algorítmica. Ele comprova numericamente que a minha decisão de usar o XGBoost como motor dos especialistas na minha arquitetura *Solarfall* é a opção mais poderosa para os dados tabulares estruturados e temporais que estou gerenciando. A sua relevância é tamanha que já o mencionei formalmente na fundamentação do meu relatório parcial para justificar por que deixei de lado abordagens com Deep Learning.

**Validação do Comportamento com *Features* Ruidosas**: O artigo comprova que árvores com gradiente (*Gradient Boosting Trees*) têm um decaimento de acurácia praticamente irrelevante ao lidar com uma grande quantidade de colunas não informativas (irregulares), suportando perfeitamente a perda de dezenas de *features* redundantes antes de degradar. Isso traz total segurança matemática para a minha abordagem agressiva de engenharia de variáveis (cálculo de aceleração, derivadas em múltiplas janelas, integrais e razões logarítmicas) seguida do "Quick Scan", confirmando que as dezenas de *features* auxiliares e cálculos cíclicos não vão "confundir" os modelos do *Solarfall*.

**Atenção ao Desbalanceamento**: O ponto fraco da metrificação deste artigo em relação ao escopo do meu projeto é a métrica base. Para uniformizar o problema, os autores forçaram que todos os conjuntos de dados tivessem classes perfeitamente balanceadas (mantendo 50% de amostras em cada classe). Eu estou lidando com um domínio físico onde as explosões representam pouco mais de 7% de toda a cronologia (com a classe extrema X figurando em apenas 0,06% dos registros). Logo, o uso sistemático da Acurácia que eles apresentam seria fatal se transposto para a minha pipeline. Por outro lado, a mecânica de aferir estabilidade reembaralhando a ordem das avaliações de hiperparâmetros (*bootstrap* em cima da busca) é um insight de validação metodológica brilhante que eu posso experimentar incorporar nas rotinas de otimização Bayesiana do Optuna.

### Relevância na Comunidade Científica

**Os autores são bem conceituados e o artigo possui quase 3k citações, que, em geral, corroboram com a ideia de que TBM são superiores a DLM para dados tabulares. Entretanto, pesquisadores de DL continuam a evoluir em algumas condições específicas**