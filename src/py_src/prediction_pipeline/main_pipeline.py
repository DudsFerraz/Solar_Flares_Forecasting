import time
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.py_src.models import SolarFlarePredictor, SolarFlarePredictionModel
from src.py_src.prediction_pipeline.ingestion import RealTimeIngestor
from src.py_src.output.output_handler import OutputHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler()]
)


def main():
    load_dotenv()
    logging.info("=== INICIANDO SISTEMA DE PREVISÃO SOLAR (V4) ===")

    windows = ['24h']

    try:
        ingestor = RealTimeIngestor()
        predictor = SolarFlarePredictor(windows=windows)
        outputs = OutputHandler()
        logging.info("Subsistemas carregados.")
    except Exception as e:
        logging.critical(f"Falha na inicialização: {e}")
        return

    last_states = {w: None for w in windows}
    LOOP_INTERVAL = 60 * 10

    while True:
        cycle_start = time.time()
        logging.info("--- Iniciando Ciclo ---")

        try:
            df_raw = ingestor.fetch_data()
            if df_raw is None or df_raw.empty:
                logging.warning("Sem dados. Retentando em 1 min.")
                time.sleep(60)
                continue

            last_data_time = df_raw.index[-1]
            latency = (datetime.now(timezone.utc) - last_data_time).total_seconds() / 60

            df_features = SolarFlarePredictionModel.generate_features(df_raw)
            last_row = df_features.iloc[-1]

            predictions = predictor.predict(last_row)

            outputs.update_dashboard_file(predictions, latency)

            for w, result in predictions.items():
                curr_class = result.get('final_class', 'Unknown')

                prev_class = last_states[w]

                if prev_class is not None and curr_class != prev_class:
                    logging.info(f"ALERTA [{w}]: {prev_class} -> {curr_class}")

                last_states[w] = curr_class

        except Exception as e:
            logging.error(f"Erro no ciclo: {e}", exc_info=True)

        elapsed = time.time() - cycle_start
        time.sleep(max(0, int(LOOP_INTERVAL - elapsed)))


if __name__ == "__main__":
    main()