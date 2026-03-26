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

        print("Calculando predições brutas...")
        pred_gk = gk.predict(X_processed)
        pred_gf = gf.predict(X_processed)
        pred_s910 = s910.predict(X_processed)

        pred_smx_log = smx.predict(X_processed)
        pred_smx_class = (pred_smx_log >= self.cutoff).astype(int)

        print("Consolidando Decisões Hierárquicas...")

        cond_no_flare = (pred_gk == 0)
        cond_ab = (pred_gk == 1) & (pred_gf == 0)
        cond_c = (pred_gk == 1) & (pred_gf == 1) & (pred_s910 == 0)
        cond_m = (pred_gk == 1) & (pred_gf == 1) & (pred_s910 == 1) & (pred_smx_class == 0)
        cond_x = (pred_gk == 1) & (pred_gf == 1) & (pred_s910 == 1) & (pred_smx_class == 1)

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

    def get_performance_dataframes(self):
        if not hasattr(self, 'results'):
            raise ValueError("Execute simulate_cascade antes.")

        df = self.results.copy()

        class_order = ["No Flare", "Class A/B", "Class C", "Class M", "Class X"]
        class_map = {label: idx for idx, label in enumerate(class_order)}

        df['Act_Num'] = df['Actual_Class'].map(class_map)
        df['Pred_Num'] = df['Predicted_Class'].map(class_map)

        metrics_data = []

        total_predictions = len(df)
        total_correct = (df['Act_Num'] == df['Pred_Num']).sum()

        for cls in class_order:
            cls_idx = class_map[cls]

            mask_real_is_cls = (df['Actual_Class'] == cls)
            mask_pred_is_cls = (df['Predicted_Class'] == cls)

            total_real = mask_real_is_cls.sum()
            total_pred = mask_pred_is_cls.sum()
            correct = (mask_real_is_cls & mask_pred_is_cls).sum()

            fn_count = (mask_real_is_cls & (df['Pred_Num'] < cls_idx)).sum()

            fp_count = (mask_pred_is_cls & (df['Act_Num'] < cls_idx)).sum()

            metrics_data.append({
                'Classe': cls,
                'Total Real': total_real,
                'Total Predito': total_pred,
                'Acertos': correct,
                'Falsos Negativos (Omissão)': fn_count,
                'Falsos Positivos (Alarme)': fp_count,
                'Precisão (%)': (correct / total_pred * 100) if total_pred > 0 else 0.0,
                'Recall (%)': (correct / total_real * 100) if total_real > 0 else 0.0
            })

        df_metrics = pd.DataFrame(metrics_data).set_index('Classe')

        df_metrics.loc['TOTAL SISTEMA'] = df_metrics.sum(numeric_only=True)
        df_metrics.loc['TOTAL SISTEMA', 'Precisão (%)'] = (total_correct / total_predictions * 100)
        df_metrics.loc['TOTAL SISTEMA', 'Recall (%)'] = (total_correct / total_predictions * 100)  # Acurácia global

        df_transitions = pd.crosstab(
            df['Actual_Class'],
            df['Predicted_Class'],
            dropna=False
        )

        df_transitions = df_transitions.reindex(index=class_order, columns=class_order, fill_value=0)

        df_transitions['Total Real'] = df_transitions.sum(axis=1)

        return df_metrics, df_transitions