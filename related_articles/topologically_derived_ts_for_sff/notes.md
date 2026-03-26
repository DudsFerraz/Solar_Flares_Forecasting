### Resumo

O artigo apresenta um modelo de previsão focado em prever explosões solares de magnitude maior que M1.0 dentro de uma janela futura de 24 horas. A abordagem principal é o uso de séries temporais derivadas de parâmetros magnéticos topológicos da fotosfera (dados SHARP do SDO/HMI), extraídos utilizando o framework ARTop. Os autores decompõem as taxas de entrada de *magnetic winding* (torção magnética) e helicidade em componentes potenciais e de transporte de corrente.

O algoritmo base escolhido foi o XGBoost. Foi realizado um extenso processo de engenharia de features que incluiu estatísticas em janelas móveis (médias, desvios padrões), *lags* temporais, suavização exponencial, curtose e, de forma muito destacada, o histórico de explosões das últimas 12 e 24 horas. Para lidar com o desbalanceamento de classes, aplicaram um fator de peso para a classe positiva (`scale_pos_weight`) diretamente no XGBoost.

Em termos de interpretabilidade, aplicaram a análise SHAP (SHapley Additive exPlanations), que confirmou o histórico de explosões e o acúmulo de helicidade/winding como as features mais determinantes para o modelo. O estudo também mapeou os principais desafios do modelo, apontando que regiões ativas com alta frequência de pequenas explosões classe C costumam enganar o preditor, gerando falsos positivos.

### Metrificação dos Resultados

A validação foi extremamente rigorosa, separando os dados não apenas temporalmente, mas garantindo que o conjunto de validação e um conjunto extra de teste cego (*holdout set*) fossem de regiões ativas completamente independentes do treino. O limite de decisão de probabilidade (*threshold*) do XGBoost não foi mantido no padrão 0.5; ele foi otimizado no conjunto de validação especificamente para maximizar o F1-Score.

As métricas utilizadas para guiar a otimização de hiperparâmetros (via busca aleatória com validação cruzada de 10 *folds* ) e avaliar a performance foram combinadas em um critério customizado:

**F1-Score**: Evidencia a habilidade do modelo em detectar o evento raro (explosão), equilibrando Precisão e Recall.

**AUC (Area Under the ROC Curve)**: Avalia a capacidade geral de discriminação entre as classes através de múltiplos limiares.

**Log-loss**: Mede a calibração probabilística do modelo, penalizando previsões muito confiantes porém erradas.

**TSS (True Skill Statistic)**: Métrica que avalia a habilidade da previsão lidando com o forte desbalanceamento.

No nível operacional de previsões diárias, o modelo atingiu um excelente TSS de 0.804 no conjunto de validação. Contudo, quando testado no *holdout set* (dados 100% inéditos), o TSS caiu para 0.524 , refletindo os desafios de generalização causados por efeitos de projeção e falsos positivos gerados por acúmulo de explosões menores.

### Relevância e Impacto para o meu projeto

**Score: 10 / 10**

**Validação Definitiva do XGBoost e da Engenharia de Features**: Assim como na arquitetura *Solarfall*, este estudo consolida o XGBoost como uma ferramenta de altíssimo desempenho para esse problema específico de dados tabulares temporais solares. Além disso, a engenharia de features que apliquei no meu projeto (médias de janelas deslizantes, derivadas temporais e histórico) é validada pelo fato de que o histórico das últimas 24h e as estatísticas móveis foram apontadas pelo SHAP deles como as variáveis de maior impacto.

**O *Specialist 910* como solução para o maior desafio deles**: Os autores relatam explicitamente que regiões com frequentes explosões classe C liberam energia e "confundem" o modelo, gerando falsos positivos para explosões de alto impacto (M/X). O meu Modelo 3 (*Specialist 910*), desenhado exclusivamente para separar 'C' de 'MX', ataca exatamente esse gargalo fisiológico dos dados solares. Isso é um argumento fortíssimo para o meu artigo científico ao justificar a existência da minha abordagem em cascata.

**Direcionamento para a próxima fase**: atributos topológicos magnéticos (Magnetic Winding e Helicidade de transporte de corrente).

**Baseline Direto**: O método deles de converter a saída probabilística otimizando o threshold via F1-score e o score de TSS de 0.524 em dados puramente isolados de *holdout*  me dão uma base de comparação quantitativa exata para avaliar o desempenho do *Solarfall*.
