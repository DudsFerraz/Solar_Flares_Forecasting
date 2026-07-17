You can switch the language using the menu below:

🌐 **Languages:** [English](README.md) | [Português](README.pt.md)

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

[📄 Read the Full Research Report Here (More info)](Presentations/relatorio_final.docx)
