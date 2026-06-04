# Solarfall: A Multi-Stage Machine Learning Architecture for Solar Flare Forecasting

Predicting solar flares is essential for mitigating space weather risks that impact critical technological infrastructures on Earth and in space. **Solarfall** is a machine learning pipeline designed to parse highly imbalanced and high-dimensional astrophysical data to predict these severe solar events. To handle the complexity of real-world space weather data, the project introduces a custom hierarchical cascade architecture built entirely with the XGBoost algorithm. This architecture progressively filters signal from noise through four specialized classification stages—ranging from a high-recall 'Gatekeeper' that flags potential anomalies to a deep 'Specialist MX' model trained specifically to isolate catastrophic, extreme-magnitude events.

## Technologies Used

* **Machine Learning Engine:** XGBoost for native class imbalance management, high-dimensional vector robustness, and tabular data scalability.
* **Data Acquisition & Web Scraping:** Automated data pipelines utilizing Selenium and FTP protocols to reliably extract operational space weather catalogs from NOAA/NCEI and JSOC/Stanford APIs.
* **Hyperparameter Optimization:** Optuna framework for the Bayesian optimization of tree structures and decision thresholds.
* **Hardware Acceleration:** CuPy library for direct VRAM allocation, enabling rapid iterative training via GPU acceleration.

## Author

**Eduardo Oliveira Ferraz de Campos** - Software Engineering Student at IFSP Câmpus São Carlos

*Advisor: Sérgio Luisir Discola Junior*

[📄 Read the Full Research Report Here (More info)](Presentations/relatorio_parcial.docx)

---

# Solarfall: Arquitetura de Machine Learning em Múltiplos Estágios para Previsão de Explosões Solares

A previsão de explosões solares (*solar flares*) é essencial para a mitigação de riscos associados ao clima espacial, que impactam diretamente infraestruturas tecnológicas críticas na Terra e no espaço. O **Solarfall** é um pipeline de aprendizado de máquina projetado para processar dados astrofísicos de alta dimensionalidade e severo desbalanceamento na previsão desses eventos. Para lidar com a complexidade dos dados reais de clima espacial, o projeto introduz uma arquitetura original em cascata hierárquica construída inteiramente com o algoritmo XGBoost. Essa arquitetura separa gradualmente o sinal do ruído através de quatro estágios de classificação especializados — desde um 'Gatekeeper' de alto *recall* que sinaliza anomalias potenciais, até um modelo profundo 'Specialist MX' treinado especificamente para isolar eventos catastróficos de magnitude extrema.

## Tecnologias Utilizadas

* **Motor de Machine Learning:** XGBoost para gerenciamento nativo de desbalanceamento de classes, robustez vetorial e escalabilidade em dados tabulares.
* **Aquisição de Dados e Web Scraping:** Pipelines de dados automatizados utilizando Selenium e protocolos FTP para extração confiável de catálogos operacionais das APIs do NOAA/NCEI e JSOC/Stanford.
* **Otimização de Hiperparâmetros:** Framework Optuna para otimização Bayesiana da estrutura das árvores e calibração de limiares de decisão.
* **Aceleração de Hardware:** Biblioteca CuPy para alocação direta de matrizes em VRAM, permitindo treinamento iterativo rápido via processamento em GPU.

## Autor

**Eduardo Oliveira Ferraz de Campos**
Estudante de Engenharia de Software no IFSP Câmpus São Carlos
*Orientador: Sérgio Luisir Discola Junior*

[📄 Leia o Relatório Completo Aqui (Mais informações)](Presentations/relatorio_parcial.docx)
