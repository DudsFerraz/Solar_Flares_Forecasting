### Resumo

O artigo de Francisco et al. (2025) foca nos limites dos modelos atuais de previsão de explosões solares, destacando especificamente como as métricas de avaliação tradicionais mascaram a incapacidade de muitos modelos de prever *mudanças* reais na atividade solar. Para contornar problemas de detecção, os autores propõem uma nova arquitetura de aprendizado profundo chamada P-CNN (Rede Neural Convolucional distribuída por *patches*). Esta abordagem utiliza imagens de disco completo (EUV do SDO/AIA e magnetogramas do SDO/HMI) para prever explosões das classes $\ge$ C e $\ge$ M em uma janela de 24 horas, sem depender da identificação prévia de Regiões Ativas (ARs), o que evita falhas de treinamento geradas por rótulos mal atribuídos.

Achei muito interessante que, embora o modelo utilize apenas rótulos de disco completo no treinamento, a distribuição em *patches* permite estimar probabilidades e posições sub-regionais para as explosões. Apesar do modelo atingir métricas que o colocam no estado da arte, o grande foco da pesquisa foi provar que esses resultados altos muitas vezes escondem o fato de que a rede possui um poder explicativo semelhante ao de um modelo de "persistência" (que apenas repete o resultado da janela anterior), falhando em superar simples palpites aleatórios durante transições de atividade.

### Metrificação dos Resultados

A validação dos resultados é o ponto central deste artigo. Os autores criticam o uso cego de métricas baseadas na Matriz de Confusão para dados altamente desbalanceados e introduzem novas formas de avaliação:

**TSS (True Skill Statistic) e HSS (Heidke Skill Score)**: O artigo demonstra matematicamente que o TSS é altamente sensível à composição do conjunto de dados e ao balanço de classes em modelos específicos. Além disso, como o TSS não incorpora informações sobre o Valor Preditivo Positivo (PPV), ele se torna incompleto e potencialmente enganoso, pois não penaliza taxas altas de alarmes falsos em cenários de extremo desbalanceamento.

**MCC (Matthews Correlation Coefficient)**: Recomendada pelos autores como uma medida muito mais robusta e agnóstica do poder explicativo do modelo. O MCC trata eventos positivos e negativos de forma simétrica e se mostrou mais estável a variações na composição dos dados durante o ciclo solar.

**Métricas Segmentadas (AC e NC)**: Esta é a inovação metodológica mais forte para validação. Eles rotularam janelas temporais como AC (*Activity Change*) se o status da explosão fosse diferente da janela anterior, e NC (*No Change*) caso a atividade se mantivesse estável. Eles recomendam avaliar o TSS, HSS e MCC estritamente nestes subconjuntos para revelar se o modelo prevê transições ou apenas "reconhece" um período que já é ativo.

**PRSS (Persistence Relative Skill Scores)**: Introduziram métricas relativas que comparam a performance do modelo diretamente contra um modelo base de persistência (que assume que a próxima janela será igual à atual), gerando um escore normalizado de -1 a 1. O **PR-F1** é destacado como a métrica mais prática e confiável, pois integra Precisão e Recall em tarefas desbalanceadas e expõe o real valor operacional do sistema.



### Relevância e Impacto para o meu projeto

**Score: 8 / 10**

**Revisão Crítica do Método de Validação**: A crítica ao TSS me alerta a não otimizar meus modelos XGBoost visando cegamente maximizar essa métrica, algo muito comum na área. É necessário investigar se essa conclusão realmente é correta e se esse artigo é relevante na bibliografia da área.

**Adoção Imediata das Métricas AC/NC e PR-F1**: A ideia de segmentar a avaliação entre janelas AC e NC pode ser aplicada diretamente no meu pipeline atual. Como já construí minhas variáveis usando janelas deslizantes e classifiquei alvos em janelas futuras (6h, 12h, 24h, 48h, 72h) , inserir um script de teste que isole as janelas onde há *mudança* de classe provará de forma contundente se a minha estratégia de *feature engineering* (como derivadas e taxas de aceleração temporal)  realmente antecipa as explosões ou se apenas performa bem durante "calmarias" e tempestades contínuas.

### Relevância na Comunidade Científica

**Encontrei duas citações, ambas corroboram com as limitações do TSS e também adotam as métricas propostas. Além disso, Teresa Barata e Dario Del Moro parecem ser autores relevantes**