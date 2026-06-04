# Solarfall: Arquitetura de Machine Learning em Múltiplos Estágios para Previsão de Explosões Solares

A previsão de explosões solares (*solar flares*) é essencial para a mitigação de riscos associados ao clima espacial, que impactam diretamente infraestruturas tecnológicas críticas na Terra e no espaço. O **Solarfall** é um pipeline de aprendizado de máquina projetado para processar dados astrofísicos de alta dimensionalidade e severo desbalanceamento na previsão desses eventos. Para lidar com a complexidade dos dados reais de clima espacial, o projeto introduz uma arquitetura original em cascata hierárquica construída inteiramente com o algoritmo XGBoost. Essa arquitetura separa gradualmente o sinal do ruído através de quatro estágios de classificação especializados — desde um 'Gatekeeper' de alto *recall* que sinaliza anomalias potenciais, até um modelo profundo 'Specialist MX' treinado especificamente para isolar eventos catastróficos de magnitude extrema.

## Tecnologias Utilizadas

* **Motor de Machine Learning:** XGBoost para gerenciamento nativo de desbalanceamento de classes, robustez vetorial e escalabilidade em dados tabulares.
* **Aquisição de Dados e Web Scraping:** Pipelines de dados automatizados utilizando Selenium e protocolos FTP para extração confiável de catálogos operacionais das APIs do NOAA/NCEI e JSOC/Stanford.
* **Otimização de Hiperparâmetros:** Framework Optuna para otimização Bayesiana da estrutura das árvores e calibração de limiares de decisão.
* **Aceleração de Hardware:** Biblioteca CuPy para alocação direta de matrizes em VRAM, permitindo treinamento iterativo rápido via processamento em GPU.

## Autor

**Eduardo Oliveira Ferraz de Campos** - Estudante de Engenharia de Software no IFSP Câmpus São Carlos
*Orientador: Sérgio Luisir Discola Junior*

[📄 Leia o Relatório Completo Aqui (Mais informações)](Presentations/relatorio_parcial.docx)
