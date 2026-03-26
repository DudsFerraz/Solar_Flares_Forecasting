from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, matthews_corrcoef, roc_auc_score, average_precision_score, f1_score
import os
pd.set_option('mode.copy_on_write', True)
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
        x_filtered = self._filter_features(x)
        probas = self.model.predict_proba(x_filtered)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probas)

        if target_recall:
            idx = np.abs(recalls - target_recall).argmin()
            if idx >= len(thresholds): idx = len(thresholds) - 1
            best_thresh = thresholds[idx]
            print(f"Threshold ajustado para Recall ~{target_recall}: {best_thresh:.4f}")
        else:
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
                            buffer_limits: Tuple[float, float] = None,
                            cutoff: float = None) -> Tuple[plt.Figure, pd.DataFrame]:

        if cutoff is not None:
            y_pred_raw = self.predict(x)
            y_true_bin = (y >= cutoff).astype(int)
            y_pred_bin = (y_pred_raw >= cutoff).astype(int)

            label_neg = f'Classe < 10^{cutoff} (Negativos)'
            label_pos = f'Classe >= 10^{cutoff} (Positivos)'
            limit_ref = 10 ** cutoff
        else:
            y_pred_bin = self.predict(x)
            y_true_bin = y
            label_neg = 'Negative Class (0)'
            label_pos = 'Positive Class (1)'
            limit_ref = None

        if buffer_limits is None:
            buffer_limits = getattr(self, 'buffer_limits', None)
        l_lim = buffer_limits[0] if buffer_limits else None
        u_lim = buffer_limits[1] if buffer_limits else None

        df_res = pd.DataFrame({
            'Flux': flux_values,
            'Truth_Bin': y_true_bin,
            'Pred_Bin': y_pred_bin
        })

        conditions = [
            (df_res['Truth_Bin'] == 1) & (df_res['Pred_Bin'] == 1),  # TP
            (df_res['Truth_Bin'] == 0) & (df_res['Pred_Bin'] == 0),  # TN
            (df_res['Truth_Bin'] == 0) & (df_res['Pred_Bin'] == 1),  # FP
            (df_res['Truth_Bin'] == 1) & (df_res['Pred_Bin'] == 0)  # FN
        ]
        choices = ['TP (Hit)', 'TN (Correct Rejection)', 'FP (False Alarm)', 'FN (Miss)']
        df_res['Outcome'] = np.select(conditions, choices, default='Error')

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        subset_neg = df_res[df_res['Truth_Bin'] == 0]
        if not subset_neg.empty:
            sns.histplot(data=subset_neg, x='Flux', hue='Outcome', multiple='stack',
                         palette={'TN (Correct Rejection)': 'lightgreen', 'FP (False Alarm)': 'red'},
                         log_scale=True, ax=axes[0], bins=50, edgecolor='black')
        axes[0].set_title(label_neg)
        if l_lim: axes[0].axvline(l_lim, color='orange', ls='--', label='Buffer Low')
        if limit_ref: axes[0].axvline(limit_ref, color='black', ls=':', label='Cutoff Class')
        axes[0].legend()

        subset_pos = df_res[df_res['Truth_Bin'] == 1]
        if not subset_pos.empty:
            sns.histplot(data=subset_pos, x='Flux', hue='Outcome', multiple='stack',
                         palette={'TP (Hit)': 'green', 'FN (Miss)': 'crimson'},
                         log_scale=True, ax=axes[1], bins=50, edgecolor='black')
        axes[1].set_title(label_pos)
        if u_lim: axes[1].axvline(u_lim, color='orange', ls='--', label='Buffer High')
        if limit_ref: axes[1].axvline(limit_ref, color='black', ls=':', label='Cutoff Class')
        axes[1].legend()

        plt.xlabel('Flux (W/m²) - Log Scale')
        plt.tight_layout()
        plt.close(fig)

        def classify_zone(row):
            f = row['Flux']
            ref_low = l_lim if l_lim else (limit_ref if limit_ref else 0)
            ref_high = u_lim if u_lim else (limit_ref if limit_ref else float('inf'))

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

        return fig, summary

    @staticmethod
    def calculate_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2 and len(np.unique(y_pred)) < 2:
            return 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr - fpr

    @staticmethod
    def calculate_hss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2 and len(np.unique(y_pred)) < 2:
            return 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total = len(y_true)
        expected_correct = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / total

        if total == expected_correct:
            return 0.0

        return (tp + tn - expected_correct) / (total - expected_correct)


    def analyze_error_distribution(self, x: DataInput, y_true: TargetInput,
                                   flux_values: pd.Series, cutoff: float = None) -> pd.DataFrame:

        if cutoff is not None:
            y_pred_raw = self.predict(x)
            y_true_bin = (y_true >= cutoff).astype(int)
            y_pred_bin = (y_pred_raw >= cutoff).astype(int)
        else:
            y_pred_bin = self.predict(x)
            y_true_bin = y_true

        df = pd.DataFrame({
            'Flux': flux_values,
            'Truth_Bin': y_true_bin,
            'Pred_Bin': y_pred_bin
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
        report_mean = report_mean.map(lambda x: f"{x:.2e}" if x > 0 else "-")
        report_mean.columns = [f"{c} Avg Flux" for c in report_mean.columns]

        final = pd.concat([report_count, report_mean], axis=1)

        order = ['A (< B1.0)', 'B (1.0 - 9.9)', 'C (1.0 - 9.9)', 'M (1.0 - 9.9)', 'X (> M10)']
        final = final.reindex([o for o in order if o in final.index])

        return final

    def get_comprehensive_metrics(self, x: DataInput, y: TargetInput) -> pd.DataFrame:
        # Conversão explícita para resolver o aviso do linter no f1_score
        y_pred = self.predict(x).astype(int)
        y_true = np.array(y).astype(int)
        y_prob = self.predict_proba(x)[:, 1]

        tss = self.calculate_tss(y_true, y_pred)
        hss = self.calculate_hss(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred) #type: ignore

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
    def calculate_prss(y_true: TargetInput, y_pred_model: TargetInput, y_pred_persistence: TargetInput) -> float:
        """
        Calcula o Persistence Relative Skill Score (PR-F1)
        y_pred_persistence: O label da janela temporal imediatamente anterior.
        """
        y_t = np.array(y_true).astype(int)
        y_p_model = np.array(y_pred_model).astype(int)
        y_p_pers = np.array(y_pred_persistence).astype(int)

        f1_model = f1_score(y_t, y_p_model) #type:ignore
        f1_persistence = f1_score(y_t, y_p_pers) #type:ignore

        if f1_model >= f1_persistence:
            pr_f1 = (f1_model - f1_persistence) / (1.0 - f1_persistence) if f1_persistence != 1.0 else 0.0
        else:
            pr_f1 = (f1_model - f1_persistence) / f1_persistence if f1_persistence != 0.0 else 0.0

        return float(pr_f1)

    def analyze_ac_nc_performance(self, x: DataInput, y_true: TargetInput, y_persistence: TargetInput) -> pd.DataFrame:
        """
        Avalia o modelo isolando as janelas de Activity Change (AC) e No Change (NC).
        """
        y_pred = self.predict(x).astype(int)
        y_t = np.array(y_true).astype(int)
        y_pers = np.array(y_persistence).astype(int)

        ac_mask = (y_t != y_pers)
        nc_mask = (y_t == y_pers)

        metrics = []

        for mask, label in zip([ac_mask, nc_mask], ["AC (Activity Change)", "NC (No Change)"]):
            if mask.sum() > 0:
                y_t_mask = y_t[mask]
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

    def save(self, filepath: str):
        joblib.dump(self, filepath)
        print(f"Modelo salvo em: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SolarFlarePredictionModel':
        return joblib.load(filepath)

    @staticmethod
    def calculate_rolling_features(df_source: pd.DataFrame, col: str,
                                   metrics_windows: list[str], deriv_windows: list[str],
                                   use_abs_for_log: bool = False,
                                   use_abs_for_ratios: bool = False) -> pd.DataFrame:

        df_col_features = pd.DataFrame(index=df_source.index)

        new_features = {}

        if use_abs_for_log:
            col_log = np.log10(np.abs(df_source[col]) + 1e-9)
        else:
            col_log = np.log10(df_source[col] + 1e-9)

        for w in metrics_windows:
            rolling = df_source[col].rolling(window=w)

            new_features[f'{col}_mean_{w}'] = rolling.mean()
            new_features[f'{col}_std_{w}'] = rolling.std()
            new_features[f'{col}_max_{w}'] = rolling.max()
            new_features[f'{col}_integ_{w}'] = rolling.sum()
            new_features[f'{col}_log_mean_{w}'] = col_log.rolling(window=w).mean()

        col_diff = df_source[col].diff()
        diff_2 = col_diff.diff()

        for w in deriv_windows:
            new_features[f'{col}_deriv_{w}'] = col_diff.rolling(w).mean()
            new_features[f'{col}_accel_{w}'] = diff_2.rolling(w).mean()

        df_col_features = pd.DataFrame(new_features, index=df_source.index)

        denom_24h = df_col_features[f'{col}_mean_24h']
        denom_7d = df_col_features[f'{col}_mean_7D']

        if use_abs_for_ratios:
            denom_24h = denom_24h.abs()
            denom_7d = denom_7d.abs()

        denom_24h = denom_24h + 1e-9
        denom_7d = denom_7d + 1e-9

        df_col_features = df_col_features.assign(**{
            f'{col}_ratio_max1h_mean24h': df_col_features[f'{col}_max_1h'] / denom_24h,
            f'{col}_ratio_max6h_mean24h': df_col_features[f'{col}_max_6h'] / denom_24h,
            f'{col}_ratio_mean24h_mean7d': df_col_features[f'{col}_mean_24h'] / denom_7d
        })

        return df_col_features

    @staticmethod
    def calculate_time_decays(target_index: pd.DatetimeIndex, events: pd.DataFrame,
                              tau_hours: float = 12.0) -> pd.DataFrame:
        """
        Calcula as features de decaimento temporal (Bdec, Cdec, Mdec, Xdec, Edec) de forma vetorizada
        e matematicamente exata sobre um grid de tempo predefinido (target_index).
        """
        decays = pd.DataFrame(index=target_index, columns=['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec']).fillna(0.0)

        if events.empty:
            return decays

        ev = events.sort_values('begin').copy()

        t_index_sec = target_index.astype(np.int64).values / 10 ** 9
        tau_sec = tau_hours * 3600.0

        bdec_arr = np.zeros(len(target_index))
        cdec_arr = np.zeros(len(target_index))
        mdec_arr = np.zeros(len(target_index))
        xdec_arr = np.zeros(len(target_index))
        edec_arr = np.zeros(len(target_index))

        ev = ev.assign(magnitude=ev['flux'].fillna(0.0))

        for _, row in ev.iterrows():
            if pd.isna(row['begin']):
                continue

            t_event_sec = row['begin'].timestamp()
            c_num = row['class_numeric']
            mag = row['magnitude']

            valid_idx = t_index_sec >= t_event_sec

            if not np.any(valid_idx):
                continue

            time_diff = t_index_sec[valid_idx] - t_event_sec
            decay_factor = np.exp(-time_diff / tau_sec)

            edec_arr[valid_idx] += mag * decay_factor

            match c_num:
                case 2:
                    bdec_arr[valid_idx] += 1.0 * decay_factor
                case 3:
                    cdec_arr[valid_idx] += 1.0 * decay_factor
                case 4:
                    mdec_arr[valid_idx] += 1.0 * decay_factor
                case _ if c_num >= 5:
                    xdec_arr[valid_idx] += 1.0 * decay_factor

        decays = decays.assign(
            Bdec=bdec_arr,
            Cdec=cdec_arr,
            Mdec=mdec_arr,
            Xdec=xdec_arr,
            Edec=edec_arr
        )

        return decays

    @staticmethod
    def generate_xray_features(xrays_to_slide: pd.DataFrame, events_to_slide: pd.DataFrame,
                               cols: list[str] = None, metrics_windows: list[str] = None,
                               deriv_windows: list[str] = None, resample_freq: str = '12min',
                               resample_method: str = 'last') -> pd.DataFrame:

        if cols is None: cols = ['xl']
        if metrics_windows is None: metrics_windows = ['1h', '6h', '12h', '24h', '7D']
        if deriv_windows is None: deriv_windows = ['5min', '15min', '30min', '1h', '3h', '6h', '12h', '24h']

        feature_dfs = []

        for col in cols:
            df_col = SolarFlarePredictionModel.calculate_rolling_features(
                xrays_to_slide, col, metrics_windows, deriv_windows,
                use_abs_for_log=False, use_abs_for_ratios=False
            )
            feature_dfs.append(df_col)

        df_features = pd.concat(feature_dfs, axis=1)

        flux_smoothed = xrays_to_slide['xl'].rolling(window='5min').mean()

        conditions = [
            (flux_smoothed >= 1e-4),  # X
            (flux_smoothed >= 1e-5),  # M
            (flux_smoothed >= 1e-6)  # C
        ]
        choices = [5, 4, 3]

        class_numeric_series = pd.Series(np.select(conditions, choices, default=0), index=xrays_to_slide.index)
        prev_class = class_numeric_series.shift(1).fillna(0)

        is_C_onset = ((class_numeric_series >= 3) & (prev_class < 3)).astype(int)
        is_M_onset = ((class_numeric_series >= 4) & (prev_class < 4)).astype(int)
        is_X_onset = ((class_numeric_series == 5) & (prev_class < 5)).astype(int)

        history_windows = ['6h', '24h', '3D', '7D']
        new_history_cols = {}

        for w in history_windows:
            new_history_cols[f'count_C_{w}'] = is_C_onset.rolling(window=w).sum()
            new_history_cols[f'count_M_{w}'] = is_M_onset.rolling(window=w).sum()
            new_history_cols[f'count_X_{w}'] = is_X_onset.rolling(window=w).sum()
            new_history_cols[f'sum_class_score_{w}'] = class_numeric_series.rolling(window=w).sum()

        df_features = df_features.assign(**new_history_cols)

        df_resampled = df_features.resample(resample_freq).agg(resample_method).ffill().dropna()

        df_decays = SolarFlarePredictionModel.calculate_time_decays(
            target_index=df_resampled.index,
            events=events_to_slide,
            tau_hours=12.0
        )

        df_final = pd.concat([df_resampled, df_decays], axis=1).dropna()

        return df_final

    @staticmethod
    def generate_target(xrays_to_slide: pd.DataFrame, events_to_slide: pd.DataFrame, target_windows: list[str] = None,
                        resample_freq: str = '12min', resample_method: str = 'last'):

        if target_windows is None:
            target_windows = ['24h']

        target_events_grouped = events_to_slide.set_index('begin')[['class_numeric', 'flux']]
        target_events_grouped = target_events_grouped.groupby(level=0).max().reindex(xrays_to_slide.index).fillna(0)

        ts_class = target_events_grouped['class_numeric']
        ts_flux = target_events_grouped['flux']

        df_target = pd.DataFrame(index=xrays_to_slide.index)
        new_target_cols = {}

        for w in target_windows:
            window_timedelta = pd.to_timedelta(w)
            window_size_int = int(window_timedelta.total_seconds() / (60 * 1))
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size_int)

            future_class_numeric_max = ts_class.rolling(window=indexer, min_periods=1).max()
            future_flux_max = ts_flux.rolling(window=indexer, min_periods=1).max()

            new_target_cols[f'target_class_in_{w}'] = (future_class_numeric_max.shift(-1).fillna(0)).astype(int)
            new_target_cols[f'target_flux_in_{w}'] = future_flux_max.shift(-1).fillna(0.0)

        df_target = df_target.assign(**new_target_cols)

        return df_target.resample(resample_freq).agg(resample_method).ffill().dropna()

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


class GreatFilterModel(SoftBufferXGBModel):
    def __init__(self, params: Dict[str, Any],
                 buffer_limits: Tuple[float, float],
                 buffer_weight: float = 0.2,
                 threshold: float = 0.5,
                 features_to_keep: List[str] = None):

        super().__init__(params, threshold, buffer_limits, buffer_weight, features_to_keep)


class Specialist910Model(SoftBufferXGBModel):
    def __init__(self, params: Dict[str, Any],
                 buffer_limits: Tuple[float, float],
                 buffer_weight: float = 0.2,
                 threshold: float = 0.5,
                 features_to_keep: List[str] = None):

        super().__init__(params, threshold, buffer_limits, buffer_weight, features_to_keep)


class SpecialistMXModel(XGBoostRegressorAdapter):
    def __init__(self, params: Dict[str, Any],
                 features_to_keep: List[str] = None,):

        super().__init__(params, threshold=0.0, features_to_keep=features_to_keep)

        def analyze_error_distribution(x, y_true, flux_values, cutoff=-4.0):
            return super().analyze_error_distribution(x, y_true, flux_values, cutoff=cutoff)

        def analyze_flux_errors(x, y, flux_values, buffer_limits=None, cutoff=-4.0):
            return super().analyze_flux_errors(x, y, flux_values, buffer_limits, cutoff=cutoff)


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
