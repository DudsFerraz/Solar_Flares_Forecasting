import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from src.py_src.models import SolarFlarePredictor


class SolarfallBacktester:
    def __init__(self, predictor: SolarFlarePredictor, window_name: str = '24h'):
        self.results = None
        self.predictor = predictor
        self.window_name = window_name
        self.cutoff = -4.0

    def simulate_cascade(self, X_processed: pd.DataFrame, y_true: pd.Series):
        print("--- Executando Simulação da Cascata (Vetorizada) ---")

        models = self.predictor.models
        gk = models['gatekeeper'].get(self.window_name)
        gf = models['great_filter'].get(self.window_name)
        s910 = models['specialist_910'].get(self.window_name)
        smx = models['specialist_mx'].get(self.window_name)

        pred_gf = None
        pred_s910 = None
        pred_smx = None

        print("1/4: Gatekeeper...")
        pred_gk = gk.predict(X_processed)
        if pred_gk is not None and pred_gk == 1:
            print("2/4: Great Filter...")
            pred_gf = gf.predict(X_processed)
        if pred_gf is not None and pred_gf == 1:
            print("3/4: Specialist 910...")
            pred_s910 = s910.predict(X_processed)
        pred_smx_log = 'no prediction'
        if pred_s910 is not None and pred_s910 == 1:
            print("4/4: Specialist MX...")
            pred_smx_log = smx.predict(X_processed)
            pred_smx = (pred_smx_log >= self.cutoff).astype(int)

        print("Consolidando Decisões...")
        cond_no_flare = (pred_gk == 0)
        cond_ab = (pred_gf is not None) & (pred_gf == 0)
        cond_c = (pred_s910 is not None) & (pred_s910 == 0)
        cond_m = (pred_smx is not None) & (pred_smx == 0)
        cond_x = (pred_smx is not None) & (pred_smx == 1)

        final_classes = np.select(
            [cond_no_flare, cond_ab, cond_c, cond_m, cond_x],
            ["No Flare", "Class A/B", "Class C", "Class M", "Class X"],
            default="Uncertain"
        )

        results = pd.DataFrame(index=X_processed.index)
        results['Predicted_Class'] = final_classes

        label_map = {0: "No Flare", 1: "Class A/B", 2: "Class A/B", 3: "Class C", 4: "Class M", 5: "Class X"}
        results['Actual_Class'] = y_true.map(label_map)
        results['Is_Correct'] = (results['Predicted_Class'] == results['Actual_Class'])

        results['Raw_GK'] = pred_gk
        results['Raw_MX_LogFlux'] = pred_smx_log

        self.results = results
        return results

    def analyze_stability(self, time_freq='7D'):
        if not hasattr(self, 'results'):
            raise ValueError("Execute simulate_cascade antes.")

        print(f"\n--- Análise de Estabilidade Temporal (Janelas de {time_freq}) ---")

        grouper = self.results.groupby(pd.Grouper(freq=time_freq))

        metrics = []

        for name, group in grouper:
            if group.empty: continue

            y_true_bin = group['Actual_Class'].isin(['Class M', 'Class X']).astype(int)
            y_pred_bin = group['Predicted_Class'].isin(['Class M', 'Class X']).astype(int)

            if y_true_bin.sum() == 0:
                recall = np.nan
            else:
                recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)

            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)

            count_x_real = (group['Actual_Class'] == 'Class X').sum()
            count_x_pred = (group['Predicted_Class'] == 'Class X').sum()

            metrics.append({
                'Period': name,
                'Records': len(group),
                'Recall_MX': recall,
                'Precision_MX': precision,
                'Count_X_Real': count_x_real,
                'Count_X_Pred': count_x_pred
            })

        df_metrics = pd.DataFrame(metrics).set_index('Period')
        return df_metrics

    def plot_stability_report(self, df_metrics):
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.plot(df_metrics.index, df_metrics['Recall_MX'], label='Recall (M+X)', color='green', marker='o')
        ax1.plot(df_metrics.index, df_metrics['Precision_MX'], label='Precision (M+X)', color='blue', linestyle='--')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_title(f'Estabilidade do Modelo Solarfall ({self.window_name}) - Análise Semanal')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.bar(df_metrics.index, df_metrics['Count_X_Real'], color='red', alpha=0.2, width=4, label='# X Reais')
        ax2.set_ylabel('Quantidade de Eventos Classe X')
        ax2.legend(loc='upper right')

        plt.show()