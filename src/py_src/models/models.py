from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

DataInput = Union[pd.DataFrame, np.ndarray]
TargetInput = Union[pd.Series, np.ndarray]


class SolarFlarePredictionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, params: Dict[str, Any], threshold: float = 0.5,
                 features_to_keep: List[str] = None):
        self.params = params
        self.threshold = threshold
        self.features_to_keep = features_to_keep
        self.model = None
        self.buffer_limits: Optional[Tuple[float, float]] = None
        self._is_fitted = False
        self._build_model()

    def _build_model(self):
        raise NotImplementedError("Subclasses devem implementar _build_model")

    def _filter_features(self, x: DataInput) -> DataInput:
        if self.features_to_keep is not None:
            if isinstance(x, pd.DataFrame):
                missing = [c for c in self.features_to_keep if c not in x.columns]
                if missing:
                    raise ValueError(f"Features obrigatórias ausentes: {missing}")
                return x[self.features_to_keep]
            else:
                pass
        return x

    def predict(self, x: DataInput) -> np.ndarray:
        x_filtered = self._filter_features(x)
        probas = self.model.predict_proba(x_filtered)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, x: DataInput) -> np.ndarray:
        x_filtered = self._filter_features(x)
        return self.model.predict_proba(x_filtered)

    def fit(self, x: DataInput, y: TargetInput, **kwargs) -> 'SolarFlarePredictionModel':
        x_filtered = self._filter_features(x)

        if 'eval_set' in kwargs:
            new_eval_set = []
            for (x_val, y_val) in kwargs['eval_set']:
                if isinstance(x_val, pd.DataFrame) and self.features_to_keep:
                    x_val = x_val[self.features_to_keep]
                new_eval_set.append((x_val, y_val))
            kwargs['eval_set'] = new_eval_set

        self.model.fit(x_filtered, y, **kwargs)
        self._is_fitted = True
        return self

    def discover_top_features(self, x: DataInput, y: TargetInput,
                              cumulative_threshold: float = 0.95,
                              flux_values: pd.Series = None) -> List[str]:

        print(f"--- Quick Scan (Discovery Mode) ---")
        fast_params = self.params.copy()
        fast_params['n_estimators'] = 300
        fast_params['learning_rate'] = 0.1

        init_kwargs = {
            'params': fast_params,
            'threshold': 0.5,
            'features_to_keep': None
        }

        if hasattr(self, 'buffer_limits'):
            init_kwargs['buffer_limits'] = getattr(self, 'buffer_limits')
            if hasattr(self, 'buffer_weight'):
                init_kwargs['buffer_weight'] = getattr(self, 'buffer_weight')

        temp_model = self.__class__(**init_kwargs)

        fit_kwargs = {}
        if flux_values is not None and hasattr(temp_model, 'buffer_limits') and temp_model.buffer_limits is not None:
            fit_kwargs['flux_values'] = flux_values

        temp_model.fit(x, y, **fit_kwargs)

        df_imp = temp_model.get_feature_importance()

        selected = df_imp[df_imp['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
        if len(selected) < 5: selected = df_imp['feature'].head(5).tolist()

        print(f"Quick Scan concluído. {len(selected)} features selecionadas (de {len(df_imp)}).")
        return selected

    def get_feature_importance(self) -> pd.DataFrame:
        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError("Modelo interno sem suporte a feature_importances_")

        importances = self.model.feature_importances_

        names = self.features_to_keep
        if names is None and hasattr(self.model, 'feature_names_in_'):
            names = self.model.feature_names_in_
        if names is None:
            names = [f"f{i}" for i in range(len(importances))]

        df = pd.DataFrame({'feature': names, 'importance': importances})
        df = df.sort_values('importance', ascending=False)
        df['cumulative_importance'] = df['importance'].cumsum()
        return df


    def optimize_threshold(self, x: DataInput, y: TargetInput, target_recall: float = None) -> float:
        """Encontra e aplica o melhor threshold."""
        x_filtered = self._filter_features(x)
        probas = self.model.predict_proba(x_filtered)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probas)

        if target_recall:
            idx = np.abs(recalls - target_recall).argmin()
            if idx >= len(thresholds): idx = len(thresholds) - 1
            best_thresh = thresholds[idx]
            print(f"Threshold ajustado para Recall ~{target_recall}: {best_thresh:.4f}")
        else:
            # Interseção P=R
            idx = np.abs(precisions[:-1] - recalls[:-1]).argmin()
            best_thresh = thresholds[idx]
            print(f"Threshold de Equilíbrio (P=R): {best_thresh:.4f}")

        self.threshold = best_thresh
        return best_thresh

    def get_threshold_graph(self, x: DataInput, y: TargetInput) -> plt.Figure:
        x_filtered = self._filter_features(x)
        probas = self.model.predict_proba(x_filtered)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probas)

        df_thresholds = pd.DataFrame({
            'Threshold': thresholds,
            'Precision (MX)': precisions[:-1],
            'Recall (MX)': recalls[:-1]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_thresholds['Threshold'], df_thresholds['Recall (MX)'], label='Recall MX (Segurança)')
        ax.plot(df_thresholds['Threshold'], df_thresholds['Precision (MX)'], label='Precision MX')
        ax.set_xlabel('Limiar de Decisão (Threshold)')
        ax.set_ylabel('Score')
        ax.set_title('Trade-off: Escolhendo o Limiar Ideal')
        ax.legend()
        ax.grid(True)
        plt.close(fig)
        return fig

    def get_classification_report(self, x: DataInput, y: TargetInput, target_names: List[str] = None) -> str:
        y_pred = self.predict(x)
        return classification_report(y, y_pred, target_names=target_names)

    def get_confusion_matrix_display(self, x: DataInput, y: TargetInput,
                                     display_labels: List[str] = None) -> ConfusionMatrixDisplay:
        y_pred = self.predict(x)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        return disp

    def analyze_flux_errors(self, x: DataInput, y: TargetInput, flux_values: pd.Series,
                            buffer_limits: Tuple[float, float] = None) -> Tuple[plt.Figure, pd.DataFrame]:
        y_pred = self.predict(x)

        if buffer_limits is None:
            buffer_limits = getattr(self, 'buffer_limits', (None, None))
            if buffer_limits is None: buffer_limits = (None, None)

        lower, upper = buffer_limits
        df_res = pd.DataFrame({'Truth': y, 'Pred': y_pred, 'Flux': flux_values})

        conditions = [
            (df_res['Truth'] == 1) & (df_res['Pred'] == 1),
            (df_res['Truth'] == 0) & (df_res['Pred'] == 0),
            (df_res['Truth'] == 0) & (df_res['Pred'] == 1),
            (df_res['Truth'] == 1) & (df_res['Pred'] == 0)
        ]
        choices = ['TP (Hit)', 'TN (Correct Rejection)', 'FP (False Alarm)', 'FN (Miss)']
        df_res['Outcome'] = np.select(conditions, choices, default='Error')

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        subset_c = df_res[df_res['Truth'] == 0]
        if not subset_c.empty:
            sns.histplot(data=subset_c, x='Flux', hue='Outcome', multiple='stack',
                         palette={'TN (Correct Rejection)': 'lightgreen', 'FP (False Alarm)': 'red'},
                         log_scale=True, ax=axes[0], bins=50, edgecolor='black')
        axes[0].set_title('Negative Class Analysis')
        if lower: axes[0].axvline(lower, color='orange', ls='--', label=f'Lower ({lower:.1e})')
        axes[0].legend()

        subset_mx = df_res[df_res['Truth'] == 1]
        if not subset_mx.empty:
            sns.histplot(data=subset_mx, x='Flux', hue='Outcome', multiple='stack',
                         palette={'TP (Hit)': 'green', 'FN (Miss)': 'crimson'},
                         log_scale=True, ax=axes[1], bins=50, edgecolor='black')
        axes[1].set_title('Positive Class Analysis')
        if upper: axes[1].axvline(upper, color='orange', ls='--', label=f'Upper ({upper:.1e})')
        axes[1].legend()

        plt.close(fig)

        def classify_zone(row):
            f = row['Flux']
            l_lim = lower if lower is not None else -float('inf')
            u_lim = upper if upper is not None else float('inf')
            if f <= l_lim: return '1. Safe Zone (Low Flux)'
            if l_lim < f < u_lim:
                return '0. No Buffer' if (lower is None and upper is None) else '2. Buffer Zone'
            return '3. Safe Zone (High Flux)'

        df_res['Zone'] = df_res.apply(classify_zone, axis=1)
        summary = df_res.groupby(['Zone', 'Outcome']).size().unstack(fill_value=0)

        if 'FP (False Alarm)' in summary.columns and 'TN (Correct Rejection)' in summary.columns:
            total_neg = summary['FP (False Alarm)'] + summary['TN (Correct Rejection)']
            summary['FP Rate (%)'] = (summary['FP (False Alarm)'] / total_neg * 100).round(1)

        if 'FN (Miss)' in summary.columns and 'TP (Hit)' in summary.columns:
            total_pos = summary['FN (Miss)'] + summary['TP (Hit)']
            summary['FN Rate (%)'] = (summary['FN (Miss)'] / total_pos * 100).round(1)

        cols = ['TN (Correct Rejection)', 'FP (False Alarm)', 'FP Rate (%)', 'TP (Hit)', 'FN (Miss)', 'FN Rate (%)']
        summary = summary.reindex(columns=[c for c in cols if c in summary.columns])
        return fig, summary

    def save(self, filepath: str):
        joblib.dump(self, filepath)
        print(f"Modelo salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SolarFlarePredictionModel':
        return joblib.load(filepath)


class XGBoostBaseAdapter(SolarFlarePredictionModel):
    def _build_model(self):
        self.model = xgb.XGBClassifier(**self.params)


class StandardXGBModel(XGBoostBaseAdapter):
    pass


class SoftBufferXGBModel(XGBoostBaseAdapter):
    def __init__(self, params: Dict[str, Any], threshold: float = 0.5,
                 buffer_limits: Optional[Tuple[float, float]] = None,
                 buffer_weight: float = 0.2,
                 features_to_keep: List[str] = None):

        self.buffer_limits = buffer_limits
        self.buffer_weight = buffer_weight
        super().__init__(params, threshold, features_to_keep)

    def fit(self, x: DataInput, y: TargetInput, flux_values: pd.Series = None, verbose: bool = True,
            **kwargs) -> 'SoftBufferXGBModel':

        if self.buffer_limits is not None:
            if flux_values is None:
                raise ValueError("SoftBufferXGBModel requer 'flux_values' no fit.")

            weights = np.ones(len(y))
            mask = (flux_values > self.buffer_limits[0]) & (flux_values < self.buffer_limits[1])
            weights[mask] = self.buffer_weight

            if verbose:
                print(f"--- Soft Buffer Training ---")
                print(f"Limites: {self.buffer_limits}")
                print(f"Peso: {self.buffer_weight} | Amostras afetadas: {np.sum(mask)}")

            if 'sample_weight' in kwargs:
                kwargs['sample_weight'] *= weights
            else:
                kwargs['sample_weight'] = weights

        super().fit(x, y, verbose=verbose, **kwargs)
        return self


class GatekeeperModel(StandardXGBModel):
    def __init__(self, params: Dict[str, Any], threshold: float = 0.5, features_to_keep: List[str] = None):
        super().__init__(params, threshold, features_to_keep)


class GreatFilterModel(StandardXGBModel):
    def __init__(self, params: Dict[str, Any], threshold: float = 0.5, features_to_keep: List[str] = None):
        super().__init__(params, threshold, features_to_keep)


class Specialist910Model(SoftBufferXGBModel):
    def __init__(self, params: Dict[str, Any],
                 buffer_limits: Tuple[float, float],
                 threshold: float = 0.5,
                 buffer_weight: float = 0.2,
                 features_to_keep: List[str] = None):
        super().__init__(params, threshold, buffer_limits, buffer_weight, features_to_keep)


class SpecialistMXModel(SoftBufferXGBModel):
    def __init__(self, params: Dict[str, Any], threshold: float = 0.5,
                 buffer_limits: Optional[Tuple[float, float]] = None,
                 buffer_weight: float = 0.2,
                 features_to_keep: List[str] = None):
        super().__init__(params, threshold, buffer_limits, buffer_weight, features_to_keep)