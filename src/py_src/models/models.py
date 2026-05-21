from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, matthews_corrcoef, roc_auc_score, average_precision_score, f1_score, fbeta_score, matthews_corrcoef
import os
import cupy as cp
# pd.set_option('mode.copy_on_write', True)
pd.set_option('future.no_silent_downcasting', True)

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
        import inspect

        print(f"--- Quick Scan (Discovery Mode) ---")
        fast_params = self.params.copy()
        fast_params['n_estimators'] = 300
        fast_params['learning_rate'] = 0.1

        init_kwargs = {
            'params': fast_params,
            'features_to_keep': None
        }

        sig = inspect.signature(self.__class__.__init__)

        if 'threshold' in sig.parameters:
            init_kwargs['threshold'] = 0.5

        if 'buffer_limits' in sig.parameters:
            init_kwargs['buffer_limits'] = getattr(self, 'buffer_limits', None)
            if 'buffer_weight' in sig.parameters:
                init_kwargs['buffer_weight'] = getattr(self, 'buffer_weight', 0.2)

        temp_model = self.__class__(**init_kwargs)

        fit_kwargs = {}
        if flux_values is not None and getattr(temp_model, 'buffer_limits', None) is not None:
            fit_kwargs['flux_values'] = flux_values

        fit_kwargs['verbose'] = False

        temp_model.fit(x, y, **fit_kwargs)

        df_imp = temp_model.get_feature_importance()

        selected = df_imp[df_imp['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
        if len(selected) < 5: selected = df_imp['feature'].head(5).tolist()

        print(f"Quick Scan concluído. {len(selected)} features selecionadas (de {len(df_imp)}).")
        return selected

    def get_feature_importance(self, plot: bool = False) -> pd.DataFrame:
        """
        Extrai a importância das features baseada no 'Gain' (Ganho de Informação).
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError("Modelo interno sem suporte a feature_importances_")

        importances = self.model.feature_importances_

        names = self.features_to_keep
        if names is None and hasattr(self.model, 'feature_names_in_'):
            names = self.model.feature_names_in_
        if names is None:
            names = [f"f{i}" for i in range(len(importances))]

        df = pd.DataFrame({'feature': names, 'importance_gain': importances})
        df = df.sort_values('importance_gain', ascending=False).copy()
        df['cumulative_importance'] = df['importance_gain'].cumsum()

        if plot and not df.empty:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance_gain', y='feature', data=df, palette='viridis')
            plt.title('Feature Importance (Gain)', fontsize=14, pad=15)
            plt.xlabel('Normalized Gain', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            plt.show()

        return df

    def optimize_threshold(self, x: DataInput, y: TargetInput, target_recall: float = None, beta: float = 3.0) -> float:
        """
        Otimiza o limiar de decisão. Se target_recall for fornecido, tenta atingi-lo,
        mas utiliza o F-beta score para arbitrar se o sacrifício na Precisão valeu a pena.
        """
        x_filtered = self._filter_features(x)
        probas = self.model.predict_proba(x_filtered)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probas)

        if target_recall:
            # 1. Busca Cega: Encontra o threshold que garante o Recall
            idx = np.abs(recalls - target_recall).argmin()
            if idx >= len(thresholds): idx = len(thresholds) - 1
            candidate_thresh = thresholds[idx]

            # 2. Arbitragem Automática (Sanity Check via F-Beta). Calcula previsões para o candidato e para o padrão (0.5)
            y_pred_cand = (probas >= candidate_thresh).astype(int)
            y_pred_def = (probas >= 0.5).astype(int)

            # F-beta pune severamente se a Precisão for destruída, mesmo com alto Recall
            fbeta_cand = fbeta_score(y, y_pred_cand, beta=beta, zero_division=0)
            fbeta_def = fbeta_score(y, y_pred_def, beta=beta, zero_division=0)

            if fbeta_cand >= fbeta_def:
                best_thresh = candidate_thresh
                print(
                    f"✅ Threshold Tuned aprovado: {best_thresh:.4f} (Recall ~{target_recall} | F{beta}={fbeta_cand:.4f})")
            else:
                best_thresh = 0.5
                print(
                    f"❌ Threshold Tuned rejeitado! A Precisão colapsou e o F{beta} ({fbeta_cand:.4f}) ficou pior que o padrão ({fbeta_def:.4f}).")
                print(f"Revertendo por segurança para o Threshold padrão: 0.5000")

        else:
            # lógica de equilíbrio caso não haja target
            idx = np.abs(precisions[:-1] - recalls[:-1]).argmin()
            best_thresh = thresholds[idx]
            print(f"Threshold de Equilíbrio (P=R): {best_thresh:.4f}")

        # 3. Salva no estado do modelo apenas o threshold que sobreviveu à arbitragem
        self.threshold = best_thresh
        return best_thresh

    def get_threshold_graph(self, x: DataInput, y: TargetInput) -> plt.Figure:
        x_filtered = self._filter_features(x)
        probas = self.model.predict_proba(x_filtered)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probas)

        df_thresholds = pd.DataFrame({
            'Threshold': thresholds,
            'Precision (1)': precisions[:-1],
            'Recall (1)': recalls[:-1]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_thresholds['Threshold'], df_thresholds['Recall (1)'], label='Recall 1 (Segurança)')
        ax.plot(df_thresholds['Threshold'], df_thresholds['Precision (1)'], label='Precision 1')
        ax.set_xlabel('Limiar de Decisão (Threshold)')
        ax.set_ylabel('Score')
        ax.set_title('Trade-off: Escolhendo o Limiar Ideal')
        ax.legend()
        ax.grid(True)
        plt.close(fig)
        return fig

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None) -> str:
        """Retorna o relatório de classificação usando vetores pré-calculados."""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def get_confusion_matrix_display(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     display_labels: List[str] = None) -> ConfusionMatrixDisplay:
        """Retorna o display da matriz de confusão usando vetores pré-calculados."""
        cm = confusion_matrix(y_true, y_pred)
        return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    @staticmethod
    def calculate_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Correção: Retorna NaN se faltar uma classe no subconjunto para não penalizar falsamente
        if len(np.unique(y_true)) < 2:
            return np.nan

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr - fpr

    @staticmethod
    def calculate_hss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Correção: Retorna NaN se faltar uma classe no subconjunto
        if len(np.unique(y_true)) < 2:
            return np.nan

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total = len(y_true)
        expected_correct = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / total

        if total == expected_correct:
            return 0.0

        return (tp + tn - expected_correct) / (total - expected_correct)

    def get_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """Calcula todas as métricas avançadas usando vetores pré-calculados."""
        tss = self.calculate_tss(y_true, y_pred)
        hss = self.calculate_hss(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            "TSS": [tss],
            "HSS": [hss],
            "MCC": [mcc],
            "ROC AUC": [roc_auc],
            "PR AUC": [pr_auc],
            "F1 Score": [f1]
        }
        return pd.DataFrame(metrics).round(4)

    @staticmethod
    def calculate_prss(y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_persistence: np.ndarray) -> float:
        """Calcula o Persistence Relative Skill Score (PR-F1)."""
        f1_model = f1_score(y_true, y_pred_model)
        f1_persistence = f1_score(y_true, y_pred_persistence)

        if f1_model >= f1_persistence:
            pr_f1 = (f1_model - f1_persistence) / (1.0 - f1_persistence) if f1_persistence != 1.0 else 0.0
        else:
            pr_f1 = (f1_model - f1_persistence) / f1_persistence if f1_persistence != 0.0 else 0.0

        return float(pr_f1)

    def analyze_ac_nc_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_persistence: np.ndarray) -> pd.DataFrame:
        """Avalia o modelo isolando as janelas de Activity Change (AC) e No Change (NC)."""
        ac_mask = (y_true != y_persistence)
        nc_mask = (y_true == y_persistence)

        metrics = []
        for mask, label in zip([ac_mask, nc_mask], ["AC (Activity Change)", "NC (No Change)"]):
            if mask.sum() > 0:
                y_t_mask = y_true[mask]
                y_p_mask = y_pred[mask]

                mcc = matthews_corrcoef(y_t_mask, y_p_mask)
                hss = self.calculate_hss(y_t_mask, y_p_mask)
                tss = self.calculate_tss(y_t_mask, y_p_mask)

                metrics.append({
                    "Subset": label,
                    "Count": mask.sum(),
                    "AC/NC-MCC": mcc,
                    "AC/NC-HSS": hss,
                    "AC/NC-TSS": tss
                })

        return pd.DataFrame(metrics).round(4)

    def analyze_flux_errors(self, y_true: np.ndarray, y_pred: np.ndarray, flux_values: pd.Series,
                            buffer_limits: Tuple[float, float] = None, plot: bool = False) -> Tuple[Any, pd.DataFrame]:
        """
        Análise de erros baseada no fluxo de raios-x.
        """
        if buffer_limits is None:
            buffer_limits = getattr(self, 'buffer_limits', None)
        l_lim = buffer_limits[0] if buffer_limits else None
        u_lim = buffer_limits[1] if buffer_limits else None

        df_res = pd.DataFrame({
            'Flux': flux_values.values if isinstance(flux_values, pd.Series) else flux_values,
            'Truth_Bin': y_true,
            'Pred_Bin': y_pred
        })

        conditions = [
            (df_res['Truth_Bin'] == 1) & (df_res['Pred_Bin'] == 1),
            (df_res['Truth_Bin'] == 0) & (df_res['Pred_Bin'] == 0),
            (df_res['Truth_Bin'] == 0) & (df_res['Pred_Bin'] == 1),
            (df_res['Truth_Bin'] == 1) & (df_res['Pred_Bin'] == 0)
        ]
        choices = ['TP (Hit)', 'TN (Correct Rejection)', 'FP (False Alarm)', 'FN (Miss)']
        df_res['Outcome'] = np.select(conditions, choices, default='Error')

        def classify_zone(row):
            f = row['Flux']
            ref_low = l_lim if l_lim else 0
            ref_high = u_lim if u_lim else float('inf')

            if f <= ref_low: return '1. Safe/Low Zone'
            if ref_low < f < ref_high: return '2. Buffer/Transition Zone'
            return '3. Danger/High Zone'

        df_res['Zone'] = df_res.apply(classify_zone, axis=1)
        summary = df_res.groupby(['Zone', 'Outcome']).size().unstack(fill_value=0)

        if 'FP (False Alarm)' in summary.columns and 'TN (Correct Rejection)' in summary.columns:
            total_neg = summary['FP (False Alarm)'] + summary['TN (Correct Rejection)']
            summary['FP Rate (%)'] = (summary['FP (False Alarm)'] / total_neg * 100).round(1)

        if 'FN (Miss)' in summary.columns and 'TP (Hit)' in summary.columns:
            total_pos = summary['FN (Miss)'] + summary['TP (Hit)']
            summary['FN Rate (%)'] = (summary['FN (Miss)'] / total_pos * 100).round(1)

        cols_order = ['TN (Correct Rejection)', 'FP (False Alarm)', 'FP Rate (%)', 'TP (Hit)', 'FN (Miss)',
                      'FN Rate (%)']
        summary = summary.reindex(columns=[c for c in cols_order if c in summary.columns])

        fig = None
        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            subset_neg = df_res[df_res['Truth_Bin'] == 0]
            if not subset_neg.empty:
                sns.histplot(data=subset_neg, x='Flux', hue='Outcome', multiple='stack',
                             palette={'TN (Correct Rejection)': 'lightgreen', 'FP (False Alarm)': 'red'},
                             log_scale=True, ax=axes[0], bins=50, edgecolor='black')
            axes[0].set_title('Negative Class (0)')
            if l_lim: axes[0].axvline(l_lim, color='orange', ls='--', label='Buffer Low')
            axes[0].legend()

            subset_pos = df_res[df_res['Truth_Bin'] == 1]
            if not subset_pos.empty:
                sns.histplot(data=subset_pos, x='Flux', hue='Outcome', multiple='stack',
                             palette={'TP (Hit)': 'green', 'FN (Miss)': 'crimson'},
                             log_scale=True, ax=axes[1], bins=50, edgecolor='black')
            axes[1].set_title('Positive Class (1)')
            if u_lim: axes[1].axvline(u_lim, color='orange', ls='--', label='Buffer High')
            axes[1].legend()

            plt.xlabel('Flux (W/m²) - Log Scale')
            plt.tight_layout()
            plt.close(fig)

        return fig, summary

    def analyze_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   flux_values: pd.Series) -> pd.DataFrame:
        """Mede a gravidade das omissões do modelo com base na classificação oficial (C, M, X)."""
        df = pd.DataFrame({
            'Flux': flux_values.values if isinstance(flux_values, pd.Series) else flux_values,
            'Truth_Bin': y_true,
            'Pred_Bin': y_pred
        })

        def get_solar_class(flux):
            if flux < 1e-7: return 'A (< B1.0)'
            if flux < 1e-6: return 'B (1.0 - 9.9)'
            if flux < 1e-5: return 'C (1.0 - 9.9)'
            if flux < 1e-4: return 'M (1.0 - 9.9)'
            return 'X (> M10)'

        df['SolarClass'] = df['Flux'].apply(get_solar_class)

        conditions = [
            (df['Truth_Bin'] == 1) & (df['Pred_Bin'] == 0),  # FN
            (df['Truth_Bin'] == 0) & (df['Pred_Bin'] == 1)  # FP
        ]
        df['ErrorType'] = np.select(conditions, ['FN (Miss)', 'FP (False Alarm)'], default='Correct')

        df_errors = df[df['ErrorType'] != 'Correct']

        if df_errors.empty:
            return pd.DataFrame(columns=['Mensagem'], data=['Nenhum erro encontrado!'])

        report_count = df_errors.pivot_table(
            index='SolarClass', columns='ErrorType', values='Flux', aggfunc='count', fill_value=0
        )
        report_mean = df_errors.pivot_table(
            index='SolarClass', columns='ErrorType', values='Flux', aggfunc='mean'
        )
        report_mean = report_mean.applymap(lambda x: f"{x:.2e}" if pd.notnull(x) and x > 0 else "-")
        report_mean.columns = [f"{c} Avg Flux" for c in report_mean.columns]

        final = pd.concat([report_count, report_mean], axis=1)
        order = ['A (< B1.0)', 'B (1.0 - 9.9)', 'C (1.0 - 9.9)', 'M (1.0 - 9.9)', 'X (> M10)']
        return final.reindex([o for o in order if o in final.index])

    def save(self, filepath: str):
        joblib.dump(self, filepath)
        print(f"Modelo salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SolarFlarePredictionModel':
        return joblib.load(filepath)


class XGBoostBaseAdapter(SolarFlarePredictionModel):
    def _build_model(self):
        self.model = xgb.XGBClassifier(**self.params)


class XGBoostRegressorAdapter(SolarFlarePredictionModel):
    def _build_model(self):
        self.model = xgb.XGBRegressor(**self.params)

    def predict(self, x: DataInput) -> np.ndarray:
        x_filtered = self._filter_features(x)
        return self.model.predict(x_filtered)

    def predict_proba(self, x: DataInput) -> np.ndarray:
        raise NotImplementedError("Modelos de regressão não possuem predict_proba. Use predict().")

    def predict_class(self, x: DataInput, cutoff_value: float) -> np.ndarray:
        y_pred_continuous = self.predict(x)
        return (y_pred_continuous >= cutoff_value).astype(int)


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

            xp = cp.get_array_module(flux_values) if 'cp' in globals() else np

            weights = xp.ones(len(y), dtype='float32')
            mask = (flux_values > self.buffer_limits[0]) & (flux_values < self.buffer_limits[1])
            weights[mask] = self.buffer_weight

            if verbose:
                print(f"--- Soft Buffer Training ---")
                print(f"Limites: {self.buffer_limits}")
                print(f"Peso: {self.buffer_weight} | Amostras afetadas: {int(mask.sum())}")

            if 'sample_weight' in kwargs:
                kwargs['sample_weight'] *= weights
            else:
                kwargs['sample_weight'] = weights

        super().fit(x, y, verbose=verbose, **kwargs)
        return self


class GatekeeperModel(StandardXGBModel):
    def __init__(self, params: Dict[str, Any], threshold: float = 0.5, features_to_keep: List[str] = None):
        super().__init__(params, threshold, features_to_keep)


class GreatFilterModel(SoftBufferXGBModel):
    def __init__(self, params: Dict[str, Any],
                 buffer_limits: Tuple[float, float] | None,
                 buffer_weight: float = 0.2,
                 threshold: float = 0.5,
                 features_to_keep: List[str] = None):

        super().__init__(params, threshold, buffer_limits, buffer_weight, features_to_keep)


class Specialist910Model(SoftBufferXGBModel):
    def __init__(self, params: Dict[str, Any],
                 buffer_limits: Tuple[float, float] | None,
                 buffer_weight: float = 0.2,
                 threshold: float = 0.5,
                 features_to_keep: List[str] = None):

        super().__init__(params, threshold, buffer_limits, buffer_weight, features_to_keep)


class SpecialistMXModel(XGBoostRegressorAdapter):
    def __init__(self, params: Dict[str, Any], features_to_keep: List[str] = None):
        super().__init__(params, threshold=0.0, features_to_keep=features_to_keep)

    def analyze_error_distribution(self, x: DataInput, y_true: np.ndarray, flux_values: pd.Series,
                                   cutoff: float = -4.0):

        y_pred_cont = self.predict(x)
        y_pred_bin = (y_pred_cont >= cutoff).astype(int)
        return super().analyze_error_distribution(y_true=y_true, y_pred=y_pred_bin, flux_values=flux_values)

    def analyze_flux_errors(self, x: DataInput, y_true: np.ndarray, flux_values: pd.Series,
                            buffer_limits: Tuple[float, float] = None, cutoff: float = -4.0, plot: bool = False):

        y_pred_cont = self.predict(x)
        y_pred_bin = (y_pred_cont >= cutoff).astype(int)
        return super().analyze_flux_errors(y_true=y_true, y_pred=y_pred_bin, flux_values=flux_values,
                                           buffer_limits=buffer_limits, plot=plot)

class SolarFlarePredictor:
    def __init__(self, windows: List[str]):
        self.windows = windows

        self.roles_env_map = {
            'gatekeeper': 'GATEKEEPER',
            'great_filter': 'GREAT_FILTER',
            'specialist_910': 'SPECIALIST_910',
            'specialist_mx': 'SPECIALIST_MX'
        }

        self.models = {role: {} for role in self.roles_env_map.keys()}

        self._load_all_models()

    def _load_all_models(self):
        for role, env_prefix in self.roles_env_map.items():
            base_path = os.getenv(f"{env_prefix}_MODELS_PATH")

            if not base_path:
                print(f"[ALERTA] Variável de ambiente {env_prefix}_MODELS_PATH não definida.")
                continue

            for w in self.windows:
                filename = f"{role}_{w}_v1.joblib"
                model_path = os.path.join(base_path, w, filename)

                if os.path.exists(model_path):
                    try:
                        model_instance = joblib.load(model_path)
                        self.models[role][w] = model_instance
                    except Exception as e:
                        print(f"[ERRO] Falha ao carregar {model_path}: {e}")
                        self.models[role][w] = None
                else:
                    print(f"[AVISO] Modelo não encontrado: {model_path}")
                    self.models[role][w] = None

    def predict(self, feature_row: pd.Series) -> Dict[str, Any]:
        x_input = feature_row.to_frame().T

        results = {}
        for w in self.windows:
            results[w] = self._predict_cascade(w, x_input)

        return results

    def _predict_cascade(self, window: str, x_input: pd.DataFrame) -> Dict[str, Any]:
        def get_prob(role_name):
            model = self.models[role_name].get(window)
            if model is None: return None

            try:
                return model.predict_proba(x_input)[:, 1][0]
            except NotImplementedError:
                return None

        def get_pred(role_name):
            model = self.models[role_name].get(window)
            if model is None: return None
            return model.predict(x_input)[0]

        gk_pred = get_pred('gatekeeper')
        gk_prob = get_prob('gatekeeper')
        if gk_pred is None:
            return {"status": "Error", "msg": "Missing Gatekeeper"}

        if gk_pred == 0:
            return {
                "final_class": "No Flare",
                "probability": 1 - gk_prob,
                "risk_level": "None",
                "path": "Gatekeeper"
            }

        gf_pred = get_pred('great_filter')
        gf_prob = get_prob('great_filter')
        if gf_pred is None:
            return {"final_class": "Potential Flare", "msg": "Missing GreatFilter"}

        if gf_pred == 0:
            return {
                "final_class": "Class A/B",
                "probability": 1 - gf_prob,
                "risk_level": "Low",
                "path": "Gatekeeper -> GreatFilter"
            }

        s910_pred = get_pred('specialist_910')
        s910_prob = get_prob('specialist_910')
        if s910_pred is None:
            return {"final_class": "Class C+", "msg": "Missing Specialist910"}

        if s910_pred == 0:
            return {
                "final_class": "Class C",
                "probability": 1 - s910_prob,
                "risk_level": "Moderate",
                "path": "Gatekeeper -> GreatFilter -> Spec910"
            }

        smx_model = self.models['specialist_mx'].get(window)
        if smx_model is None:
            return {"final_class": "Class M+", "msg": "Missing SpecialistMX"}

        log_flux_pred = smx_model.predict(x_input)[0]

        cutoff_x = -4.0

        k = 10
        pseudo_prob_x = 1 / (1 + np.exp(-k * (log_flux_pred - cutoff_x)))

        if log_flux_pred < cutoff_x:
            return {
                "final_class": "Class M",
                "probability": 1 - pseudo_prob_x,
                "estimated_flux": 10 ** log_flux_pred,
                "risk_level": "High",
                "path": "Gatekeeper -> GreatFilter -> Spec910 -> SpecMX"
            }
        else:
            return {
                "final_class": "Class X",
                "probability": pseudo_prob_x,
                "estimated_flux": 10 ** log_flux_pred,
                "risk_level": "Extreme",
                "path": "Gatekeeper -> GreatFilter -> Spec910 -> SpecMX"
            }


class SolarfallFeatureEngineer:
    def __init__(self, freq='12min'):
        self.freq = freq
        self.steps_per_hour = 60 // 12

    def _apply_stride(self, df: pd.DataFrame, group_cols: list, dropna_subset: list) -> pd.DataFrame:
        """
        Aplica o stride (avanço de 1h) e expurga o período de burn-in (linhas com NaN nas features de 24h).
        O agrupamento garante que o pulo respeite as fronteiras de continuidade.
        """
        # Fatiamento vetorizado: pega 1 linha a cada 5 (1h) dentro do bloco contínuo
        df_strided = df.groupby(group_cols, group_keys=False).apply(
            lambda x: x.iloc[::self.steps_per_hour]
        ).reset_index(drop=True)

        # Hard Drop focado: remove apenas linhas onde as features de limite temporal (burn-in) falharam
        return df_strided.dropna(subset=dropna_subset)

    def generate_family_a_xray(self, df_xray: pd.DataFrame) -> pd.DataFrame:
        """Família A: Série Temporal Global (Raio-X) agrupada estritamente por run_id."""
        # 1. LIMPEZA ABSOLUTA DE NULOS CRÍTICOS
        df = df_xray.dropna(subset=['run_id', 'time']).copy()

        df = df.sort_values(['run_id', 'time']).reset_index(drop=True)
        gb = df.groupby('run_id')

        # 2. Estatísticas de Estado
        windows = ['1h', '6h', '12h', '24h']
        for w in windows:
            roll = gb.rolling(window=w, on='time')['xrsb_flux_mean']
            df[f'xray_mean_{w}'] = roll.mean().to_numpy()
            df[f'xray_max_{w}'] = roll.max().to_numpy()
            df[f'xray_std_{w}'] = roll.std().to_numpy()

        # 3. Derivadas Temporais (Absolutas)
        df['xray_delta_1h'] = (
                    df['xrsb_flux_mean'] - gb['xrsb_flux_mean'].shift(1 * self.steps_per_hour)).abs().to_numpy()
        df['xray_delta_6h'] = (
                    df['xrsb_flux_mean'] - gb['xrsb_flux_mean'].shift(6 * self.steps_per_hour)).abs().to_numpy()
        df['xray_delta_12h'] = (
                    df['xrsb_flux_mean'] - gb['xrsb_flux_mean'].shift(12 * self.steps_per_hour)).abs().to_numpy()

        # 4. Frequência Histórica (Detecção de Onsets vetorizada por run_id)
        conditions = [
            (df['xrsb_flux_max'] >= 1e-4),  # X
            (df['xrsb_flux_max'] >= 1e-5),  # M
            (df['xrsb_flux_max'] >= 1e-6)  # C
        ]
        choices = [5, 4, 3]

        df['flare_class_num'] = np.select(conditions, choices, default=0)

        gb = df.groupby('run_id')

        df['prev_class'] = gb['flare_class_num'].shift(1).fillna(0).to_numpy()

        df['is_C_onset'] = ((df['flare_class_num'] >= 3) & (df['prev_class'] < 3)).astype(int)
        df['is_M_onset'] = ((df['flare_class_num'] >= 4) & (df['prev_class'] < 4)).astype(int)
        df['is_X_onset'] = ((df['flare_class_num'] == 5) & (df['prev_class'] < 5)).astype(int)

        hist_windows = ['24h', '72h']
        for w in hist_windows:
            df[f'count_C_{w}'] = gb.rolling(window=w, on='time')['is_C_onset'].sum().to_numpy()
            df[f'count_M_{w}'] = gb.rolling(window=w, on='time')['is_M_onset'].sum().to_numpy()
            df[f'count_X_{w}'] = gb.rolling(window=w, on='time')['is_X_onset'].sum().to_numpy()

        drop_cols = ['flare_class_num', 'prev_class', 'is_C_onset', 'is_M_onset', 'is_X_onset']
        df = df.drop(columns=drop_cols)

        df['burn_in_marker_24h'] = gb['xrsb_flux_mean'].shift(24 * self.steps_per_hour).to_numpy()

        df = self._apply_stride(df, group_cols=['run_id'], dropna_subset=['burn_in_marker_24h'])
        return df.drop(columns=['burn_in_marker_24h'])

    def generate_family_b_mag_global(self, df_mag: pd.DataFrame) -> pd.DataFrame:
        """Família B: Mag Global (Identificação dinâmica de Gaps)."""
        # 1. LIMPEZA ABSOLUTA
        df = df_mag.dropna(subset=['T_REC_round']).copy()

        df = df.sort_values('T_REC_round').reset_index(drop=True)

        delta_t = df['T_REC_round'].diff().dt.total_seconds() / 60.0
        mask_new_run = (delta_t > 15.0) | delta_t.isna()

        # Força o array NumPy para silenciar o warning do Pandas
        df['global_run_id'] = mask_new_run.cumsum().to_numpy()

        gb = df.groupby('global_run_id')
        base_cols = ['USFLUX_SUM', 'TOTUSJH_SUM', 'USFLUX_MAX', 'R_VALUE_MAX']

        for w in ['12h', '24h']:
            for col in base_cols:
                roll = gb.rolling(window=w, on='T_REC_round')[col]
                df[f'{col}_mean_{w}'] = roll.mean().to_numpy()
                df[f'{col}_max_{w}'] = roll.max().to_numpy()

        for col in base_cols:
            df[f'{col}_delta_6h'] = (df[col] - gb[col].shift(6 * self.steps_per_hour)).to_numpy()
            df[f'{col}_delta_24h'] = (df[col] - gb[col].shift(24 * self.steps_per_hour)).to_numpy()

        return self._apply_stride(df, group_cols=['global_run_id'], dropna_subset=['USFLUX_SUM_delta_24h'])

    def generate_family_c_mag_regional(self, df_mag_reg: pd.DataFrame) -> pd.DataFrame:
        """Família C: Mag Regional isolado por Região Física e Continuidade, com Densidades Magnéticas."""
        # 1. LIMPEZA ABSOLUTA (Região, Run e Tempo não podem ser nulos)
        df = df_mag_reg.dropna(subset=['REGION_ID', 'run_id', 'T_REC_round']).copy()

        # 2. ENGENHARIA FÍSICA: VARIÁVEIS INTENSIVAS (DENSIDADES)
        # Adicionamos epsilon (1e-8) para evitar erro de divisão por zero caso alguma área venha zerada
        epsilon = 1e-8
        df['DENS_TOTUSJH'] = df['TOTUSJH'] / (df['AREA_ACR'] + epsilon)
        df['DENS_TOTPOT'] = df['TOTPOT'] / (df['AREA_ACR'] + epsilon)
        df['DENS_USFLUX'] = df['USFLUX'] / (df['AREA_ACR'] + epsilon)
        df['DENS_TOTUSJZ'] = df['TOTUSJZ'] / (df['AREA_ACR'] + epsilon)

        df = df.sort_values(['REGION_ID', 'run_id', 'T_REC_round']).reset_index(drop=True)
        gb = df.groupby(['REGION_ID', 'run_id'])

        base_cols = [
            'USFLUX', 'MEANSHR', 'TOTUSJH', 'DENS_TOTUSJH', 'DENS_TOTPOT', 'DENS_USFLUX', 'DENS_TOTUSJZ'
        ]

        for col in base_cols:
            roll = gb.rolling(window='24h', on='T_REC_round')[col]
            df[f'{col}_max_24h'] = roll.max().to_numpy()

            df[f'{col}_delta_6h'] = gb[col].diff(periods=6 * self.steps_per_hour).to_numpy()
            df[f'{col}_delta_24h'] = gb[col].diff(periods=24 * self.steps_per_hour).to_numpy()

        return self._apply_stride(df, group_cols=['REGION_ID', 'run_id'], dropna_subset=['USFLUX_delta_24h'])

    def append_targets(self, df_features: pd.DataFrame, df_events: pd.DataFrame,
                       time_col: str, window_hours: int = 24,
                       is_regional: bool = False, regional_col_feature: str = 'REGION_ID',
                       harp_to_noaa_map: dict = None) -> pd.DataFrame:
        """
        Acopla as variáveis alvo (Target) ao DataFrame de features já stridado.
        Utiliza busca binária (O(N log N)) para máxima performance.
        Se for regional, requer um dicionário de mapeamento HARPNUM -> [NOAA_ARs].
        """
        df = df_features.copy()

        # 1. Preparação da matriz de eventos
        ev = df_events.dropna(subset=['peak_time']).copy()
        class_map = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}

        # Converte para array nativo para silenciar warnings
        ev['class_num'] = ev['flare_class'].str[0].map(class_map).fillna(0).astype(int).to_numpy()
        ev['flux'] = (10 ** ev['log10_intensity']).to_numpy()

        # Garante ordenação temporal para a busca binária
        ev = ev.sort_values('peak_time')

        # 2. Extração para arrays NumPy
        times_features = df[time_col].to_numpy(dtype='datetime64[ns]')
        times_events = ev['peak_time'].to_numpy(dtype='datetime64[ns]')
        classes_events = ev['class_num'].to_numpy(dtype=int)
        fluxes_events = ev['flux'].to_numpy(dtype=float)

        target_classes = np.zeros(len(df), dtype=int)
        target_fluxes = np.zeros(len(df), dtype=float)

        window_ns = np.timedelta64(window_hours, 'h')

        # 3. Mapeamento Global (Famílias A e B)
        if not is_regional:
            left_idxs = np.searchsorted(times_events, times_features, side='right')
            right_idxs = np.searchsorted(times_events, times_features + window_ns, side='right')

            for i in range(len(df)):
                l, r = left_idxs[i], right_idxs[i]
                if l < r:
                    target_classes[i] = np.max(classes_events[l:r])
                    target_fluxes[i] = np.max(fluxes_events[l:r])

        # 4. Mapeamento Regional com Tradução (Família C)
        else:
            if harp_to_noaa_map is None:
                raise ValueError("Para mapeamento regional, o dicionário harp_to_noaa_map deve ser fornecido.")

            regions_events = ev['active_region_no'].to_numpy(dtype=float, na_value=np.nan)
            regions_features = df[regional_col_feature].to_numpy(dtype=float, na_value=np.nan)

            left_idxs = np.searchsorted(times_events, times_features, side='right')
            right_idxs = np.searchsorted(times_events, times_features + window_ns, side='right')

            for i in range(len(df)):
                l, r = left_idxs[i], right_idxs[i]
                if l < r:
                    current_harp = regions_features[i]

                    if not np.isnan(current_harp):
                        # Vai buscar a lista de NOAAs (pode ter 1 ou várias) que pertencem a este HARP
                        valid_noaas = harp_to_noaa_map.get(int(current_harp), [])

                        # Verifica se as NOAAs das explosões desta janela estão dentro das NOAAs válidas
                        region_mask = np.isin(regions_events[l:r], valid_noaas)

                        if np.any(region_mask):
                            target_classes[i] = np.max(classes_events[l:r][region_mask])
                            target_fluxes[i] = np.max(fluxes_events[l:r][region_mask])

        # 5. Acoplamento final
        df = df.assign(**{
            f'target_class_in_{window_hours}h': target_classes,
            f'target_flux_in_{window_hours}h': target_fluxes
        })

        return df
