# FERRAMENTAS

---

## `Pandas`

#### A principal biblioteca para manipulação e análise de dados tabulares e séries temporais em Python. Extremamente poderosa e flexível para manipulação de dados, com excelente suporte a séries temporais, ampla documentação e comunidade gigantesca. Será a nossa ferramenta de trabalho diária.

**Útil para:**
* Realizar cálculos matemáticos rápidos em grandes volumes de dados.
* Limpeza e preparação de dados.
* Análise exploratória de dados.

---

## `Numpy`

A biblioteca fundamental para computação numérica em Python. Fornece o objeto `array`, que é muito mais eficiente para operações matemáticas do que as listas Python.

**Útil para:**
* Realizar cálculos matemáticos rápidos em grandes volumes de dados.
* Operações com matrizes e álgebra linear.

---

## `Matplotlib`

A biblioteca mais tradicional e amplamente utilizada para a criação de gráficos e visualizações estáticas em Python.

**Útil para:**
* Plotar as séries temporais para identificar tendências e anomalias.
* Criar gráficos de barras, histogramas, gráficos de dispersão, etc.
* Visualizar os resultados dos modelos.





# MODELAGEM ESTATÍSTICA & ML CLÁSSICO

---

## `SciKit-Learn`

A biblioteca padrão para machine learning em Python. Oferece uma vasta gama de algoritmos de pré-processamento, classificação, regressão e clusterização.

**Útil para:**
* Modelos: Treinar classificadores como RandomForestClassifier, GradientBoostingClassifier, LogisticRegression e SVC.
* Pré-processamento: Normalizar features com StandardScaler ou MinMaxScaler.
* Validação: Usar TimeSeriesSplit para validação cruzada, evitando usar dados do futuro para treinar o modelo que prevê o passado (um erro comum).
* Métricas de Avaliação: Calcular métricas cruciais para problemas desbalanceados como precision, recall, f1-score, roc_auc_score e plotar a matriz de confusão.

**Pontos Positivos:**
* API extremamente consistente e fácil de usar.
* Documentação excelente e vasta comunidade.
* Ampla seleção de algoritmos robustos e testados.

**Pontos Negativos:**
* Não é "nativa" para séries temporais. Requer engenharia de features (lags, médias móveis) manualmente com Pandas antes de passar os dados para os modelos.

---

## `StatsModels`

#### Uma biblioteca focada em modelos estatísticos e econométricos. É mais rigorosa do ponto de vista estatístico do que o SciKit-Learn.

**Útil para:**
* Análise e entendimento dos dados.
* Análise de Séries Temporais: Ferramentas para testes de estacionariedade, decomposição da série em tendência, sazonalidade e resíduo.
* Funções de Correlação: Gerar os gráficos ACF e PACF (plot_acf, plot_pacf) para entender a estrutura de dependência temporal dos dados.

---

## `ImbLearn`
####  Uma biblioteca complementar ao SciKit-Learn, criada especificamente para lidar com datasets desbalanceados.

**Útil para:**
* Lidar com datasets desbalanceados.
* Técnicas de Reamostragem: Implementar algoritmos como SMOTE (para criar exemplos sintéticos da classe minoritária - "sim") ou RandomUnderSampler (para remover exemplos da classe majoritária - "não").

**Pontos Positivos:**
* Integração: Funciona perfeitamente com os pipelines do SciKit-Learn.
* Solução essencial e fácil de usar para o problema do desbalanceamento.

**Pontos Negativos:**
* A aplicação incorreta de reamostragem pode levar a uma superestimação do desempenho do modelo. Deve ser aplicada apenas no conjunto de treinamento.

---

# BIBLIOTECAS ESPECIALIZADAS EM SÉRIES TEMPORAIS

---

## `Prophet`
####  Uma biblioteca Python do Facebook para previsão de séries temporais univariadas.

**Pontos Positivos:**
* Muito fácil de usar, robusta a dados faltantes e outliers.
* Ótima para séries com sazonalidades de padrão "humano" (semanal, anual).

**Pontos Negativos:**
* Não é projetada para classificação nem para dados multivariados.
* Os ciclos solares não seguem padrões humanos, o que limita a eficácia do Prophet no nosso caso.

---

## `SkTime`
####  Um framework unificado para aprendizado de máquina com séries temporais, inspirado no SciKit-Learn.

**Pontos Positivos:**
* Pipeline Unificado: Oferece uma estrutura para todas as etapas: transformação de dados, engenharia de features e modelagem de séries temporais.
* Classificadores Nativos: Inclui algoritmos de classificação de séries temporais que usam a série inteira como entrada, em vez de apenas valores pontuais (ex: TimeSeriesForestClassifier).
* Possui ferramentas de validação cruzada temporal já implementadas, evitando erros comuns.
* Ampla gama de ferramentas específicas para séries temporais.

**Pontos Negativos:**
* Comunidade menor e curva de aprendizado um pouco maior que a do SciKit-Learn.

---

## `Python Darts`
#### Uma biblioteca moderna que unifica modelos clássicos e de deep learning para previsão de séries temporais.

**Pontos Positivos:**
* Ampla Gama de Modelos: Permite treinar e comparar facilmente ARIMA, Suavização Exponencial, Prophet e modelos de Deep Learning (LSTM, Transformers) com uma única API.
* Suporte Multivariado: Lida bem com múltiplas séries temporais (raios-x e magnetogramas) como entrada.
* Backtesting Fácil: Ótimas ferramentas para simular previsões históricas e avaliar o desempenho do modelo.
* Excelente para comparar rapidamente o desempenho de diferentes classes de modelos.
* API muito amigável e moderna.

**Pontos Negativos:**
* O foco principal é em previsão (regressão). Para problemas de classificação, a abordagem consiste em modelar a probabilidade do evento, exigindo uma etapa final de decisão baseada em um limiar (ex: se P > 0.5, então classe = "sim").

---

# FRAMEWORKS BASE PARA DEEP LEARNING
### Antes das bibliotecas de alto nível, existem os frameworks fundamentais que fornecem os blocos de construção para qualquer modelo de rede neural. Eles oferecem máxima flexibilidade ao custo de uma maior complexidade e são essenciais para implementar arquiteturas de modelos personalizadas ou de última geração.

---

## `PyTorch`
#### Um framework de código aberto para machine learning, desenvolvido e mantido principalmente pela Meta AI. É amplamente conhecido por sua simplicidade e por ter uma API que se integra de forma muito natural com o ecossistema Python, o que o tornou extremamente popular na comunidade de pesquisa.

**Pontos Positivos:**
* Ideal para implementar arquiteturas de modelos encontradas em artigos científicos recentes, já que muitos pesquisadores lançam seus códigos em PyTorch.
* API "Pythonica": A sua sintaxe é muito intuitiva e segue os padrões da programação orientada a objetos do Python, facilitando a curva de aprendizado.
* Grafo Computacional Dinâmico: Permite que a estrutura da rede neural possa ser modificada em tempo de execução, o que simplifica enormemente o processo de depuração.
* Forte Adoção na Academia: É a ferramenta preferida em muitas universidades e centros de pesquisa, garantindo acesso rápido a novas arquiteturas e modelos.
* Ecossistema Robusto: Bibliotecas de ponta, como a Hugging Face Transformers, são construídas primariamente sobre PyTorch.

**Pontos Negativos:**
* Embora tenha melhorado drasticamente com ferramentas como o TorchServe, historicamente o TensorFlow era considerado mais robusto para a implantação de modelos em ambientes de produção em larga escala.

---

## `TensorFlow`
#### Um framework de código aberto para machine learning, desenvolvido e mantido pelo Google Brain. Foi projetado desde o início com foco em escalabilidade e implantação em produção, sendo uma escolha muito forte para projetos de grande porte.

**Pontos Positivos:**
* Construção de Pipelines de Produção: Oferece um ecossistema completo (TensorFlow Extended - TFX) para gerenciar o ciclo de vida do modelo, desde a ingestão dos dados até a implantação e monitoramento.
* Treinamento Distribuído: Facilita o treinamento de modelos massivos em múltiplos servidores ou em hardware especializado como as TPUs (Tensor Processing Units) do Google.
* Ecossistema de Produção Maduro: Ferramentas como TensorFlow Serving e TensorFlow Lite (para dispositivos móveis e embarcados) são extremamente robustas e otimizadas para performance.
* TensorBoard: Sua ferramenta de visualização nativa, o TensorBoard, é excelente para monitorar o treinamento e analisar o comportamento dos modelos.
* Escalabilidade: É a principal referência para treinamento de modelos em clusters de grande escala.
* Keras API: A integração nativa com o Keras como sua API de alto nível tornou a construção de modelos em TensorFlow muito mais simples e acessível.

**Pontos Negativos:**
* Curva de Aprendizado: Embora o Keras ajude, a API de baixo nível do TensorFlow pode ser menos intuitiva que a do PyTorch, especialmente para quem está começando.
* Flexibilidade: A abordagem de grafo computacional estático (em versões mais antigas) tornava a depuração mais complexa, embora o "eager execution" no TF2.x tenha resolvido grande parte disso, aproximando-o do PyTorch.

---

# FRAMEWORKS DE DEEP LEARNING & AUTOML

---

## `GluonTS`
####  Um toolkit, em Python, da Amazon (baseado em PyTorch/MXNet) para previsão de séries temporais com modelos de deep learning, com foco em previsão probabilística.

**Pontos Positivos:**
* Permite usar modelos de ponta: Acesso a implementações de modelos como DeepAR e Transformers, que podem capturar padrões temporais muito complexos.
* Previsão Probabilística: Em vez de prever "sim/não", o modelo pode prever a distribuição de probabilidade de uma explosão, o que é muito mais informativo.
* Modelos de alta performance e escaláveis para grandes datasets.

**Pontos Negativos:**
* Requer conhecimento de deep learning.
* Computacionalmente mais intensivo para treinar.

---

## `AutoGluon`
#### Uma biblioteca Python de AutoML (Machine Learning Automatizado) da Amazon que simplifica a criação de modelos de alta performance. Possui um módulo específico, AutoGluon-TimeSeries.

**Pontos Positivos:**
* Esforço mínimo: Input dos dados e ele automaticamente testa uma vasta gama de modelos (desde os simples até ensembles de redes neurais) e entrega o melhor.
* Excelente para criar um baseline muito forte rapidamente.

**Pontos Negativos:**
* É uma "caixa-preta", dificultando a interpretação do modelo final.
* Pode ser computacionalmente pesado e demorado.

---

## `TSMixer`
#### Não é uma biblioteca, mas sim uma arquitetura de modelo recente proposta pelo Google. É um modelo baseado em Redes Neurais de Múltiplas Camadas (MLPs) que desafia a complexidade dos modelos baseados em Transformers, mostrando resultados competitivos.

**Pontos Positivos:**
* Potencialmente mais simples e computacionalmente mais eficiente que modelos Transformers.
* Resultados de ponta em benchmarks.

**Pontos Negativos:**
* Requer implementação manual: Necessário implementar a arquitetura TSMixer usando um framework como PyTorch ou TensorFlow, envolvendo conhecimento avançado de deep learning.
* Sendo muito novo, há menos exemplos e suporte da comunidade.

---

## `Lag-Llama`
#### Um Foundation Model para previsão de séries temporais, proposto por pesquisadores da ServiceNow Research. O modelo adapta a arquitetura do Llama, um LLM da Meta, para o domínio de dados temporais. Foi pré-treinado em um vasto e diverso conjunto de dados de séries temporais de código aberto.

**Pontos Positivos:**
* Representa uma nova abordagem na qual um único modelo massivo pode ser aplicado a diversas tarefas de previsão com pouco ou nenhum treinamento adicional (previsão zero-shot).
* Generalização: Potencial para capturar padrões complexos e universais em dados temporais que modelos treinados apenas em um domínio específico (como dados solares) podem não ver.

**Pontos Negativos:**
* Experimental: É uma tecnologia muito nova, e sua aplicabilidade a domínios altamente especializados como a física solar ainda é uma área de pesquisa ativa.
* Complexidade e Recursos: Utilizar e, principalmente, fazer o ajuste fino (fine-tuning) desses modelos pode exigir recursos computacionais significativos.
* Foco em Regressão: Assim como o Darts, seu foco é prever valores futuros, exigindo adaptação para o problema de classificação.

---