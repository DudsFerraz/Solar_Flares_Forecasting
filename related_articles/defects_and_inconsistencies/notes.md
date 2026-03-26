### Resumo

"Defects and Inconsistencies in Solar Flare Data Sources: Implications for Machine Learning Forecasting" (Hu et al., 2026) investiga como diferentes fontes e qualidades de dados de explosões solares afetam o desempenho preditivo de modelos de Machine Learning. Os autores comparam as diferenças entre dados operacionais e dados processados retroativamente ("Science-Quality") provenientes dos satélites GOES da NOAA. Eles treinaram modelos de *Deep Learning* (LSTM) e estatísticos (Regressão Logística) para a classificação de explosões fortes (M/X) contra explosões fracas (A/B) e observaram como as inconsistências nos catálogos de eventos e nos preditores alteram os resultados das previsões dependendo da fase do ciclo solar.

### Como os Resultados Foram Metrificados e Validados
O desempenho dos modelos foi medido através de um conjunto de seis métricas derivadas da matriz de confusão:


**TSS (True Skill Statistic):** Avaliada ao longo de um grid de limiares (thresholds) probabilísticos, onde o modelo buscava o limiar ótimo que maximizasse essa métrica específica na validação.


**HSS (Heidke Skill Score), POD (Probability of Detection / Recall), F1, FAR (False Alarm Rate) e ACC (Accuracy):** Foram reportadas em conjunto. É interessante notar que a discussão dos autores aborda a necessidade de identificar explosões corretamente sem elevar drasticamente os falsos alarmes (FAR).

**Análise por Fases de Atividade:** Para uma compreensão justa, os resultados de teste foram desmembrados nas diferentes fases de atividade solar: "Mínima" (2020-2021 com extremo desbalanceamento), "Em Evolução" (2022) e "Máxima" (2023-2024, apresentando alta frequência de eventos).

**Bootstrap Ensemble:** Para garantir que os resultados fossem robustos à variabilidade da amostragem, aplicou-se uma estratégia de *bootstrap* com 30 repetições com substituição a partir do conjunto de treino, reportando a média e o intervalo dos resultados alcançados.

### Relevância e Impacto para o meu Projeto

**Score : 9.5 / 10**

**Justificativa do Impacto:**
Este artigo de 2026 é uma "bússola" para os problemas de dados enfrentados na fase de coleta.

**Atenção Crítica aos Dados Brutos:** O artigo expõe que os dados operacionais antigos do GOES (antes do GOES-16, em dezembro de 2019) possuíam um "fator de escala" (scaling factor) aplicado pela SWPC que tornava as classificações de intensidade inconsistentes com as mais recentes; uma explosão antes do GOES-R precisava ser 42% maior fisicamente para receber a mesma magnitude nas tabelas. Como sua base compila dados mensais de raios-X de 1996 a 2024, este fenômeno relatado na pesquisa afeta diretamente a homogeneidade das suas variáveis alvo.

**Near Limbs low quality:** Afirma que regiões próximas à borda do disco solar, (longitude > 70°) sofrem com erros de projeção e degradação de qualidade.

**Mapeamento de Regiões Ativas Solares** O artigo traz soluções para mapear as regiões ativas HMI SHARP---NOOA GOES

### Relevância na Comunidade Científica

**O artigo é muito recente e por isso ainda não possui citações, porém os autores são extremammente bem conceituados**