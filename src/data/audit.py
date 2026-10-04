"""Streaming audit of the PhysioNet/CinC 2019 training files.

Statistics below are exact unless a field name says reservoir. Missing entries
stay missing. No row is dropped because a clinical value is NaN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import read_psv, validate_header
from src.data.schema import (
    ALL_SPECS,
    CLINICAL_MEASUREMENTS,
    COLUMN_BY_NAME,
    EXPECTED_COLUMNS,
    FIO2_NAME,
    GROUP_A_RAPID,
    GROUP_B_SPARSE,
    GROUP_C_STATIC,
    TARGET,
    TRAJECTORY_VARIABLES,
)
from src.features.temporal import candidate_feature_names
from src.labels.targets import DT_OPTIMAL, label_is_monotonic_nondecreasing
from src.preprocessing.imputation import empirical_frequency_class, recommend_representation

MAX_GAP_HOURS = 2000
RESERVOIR_SIZE = 20000
RESERVOIR_SEED = 2019
TRAJECTORY_MIN_OFFSET = -48
TRAJECTORY_MAX_OFFSET = 0
ICULOS_CURVE_MAX = 168
COMPARE_VARIABLES = ("HR", "MAP", "Resp", "Lactate")
HOSPITALS = ("A", "B")


@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        n_batch = int(values.size)
        if n_batch == 0:
            return
        mean_batch = float(values.mean())
        m2_batch = float(np.sum((values - mean_batch) ** 2))
        batch_min = float(values.min())
        batch_max = float(values.max())
        if self.n == 0:
            self.n = n_batch
            self.mean = mean_batch
            self.m2 = m2_batch
            self.minimum = batch_min
            self.maximum = batch_max
            return
        total = self.n + n_batch
        delta = mean_batch - self.mean
        self.mean = self.mean + delta * n_batch / total
        self.m2 = self.m2 + m2_batch + delta * delta * self.n * n_batch / total
        self.n = total
        self.minimum = min(self.minimum, batch_min)
        self.maximum = max(self.maximum, batch_max)

    @property
    def std(self) -> float:
        if self.n < 2:
            return float("nan")
        return float(np.sqrt(self.m2 / (self.n - 1)))


class _Reservoir:
    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self.size = size
        self.rng = rng
        self.sample = np.empty(size, dtype=np.float64)
        self.seen = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        n = int(values.size)
        if n == 0:
            return
        if self.seen < self.size:
            take = min(self.size - self.seen, n)
            self.sample[self.seen : self.seen + take] = values[:take]
            self.seen += take
            values = values[take:]
            n = int(values.size)
            if n == 0:
                return
        draws = self.rng.integers(0, self.seen + np.arange(1, n + 1))
        mask = draws < self.size
        if np.any(mask):
            self.sample[draws[mask]] = values[mask]
        self.seen += n

    def percentiles(self, qs: tuple[float, ...]) -> list[float | None]:
        if self.seen <= 0:
            return [None for _ in qs]
        data = self.sample if self.seen >= self.size else self.sample[: self.seen]
        return [float(value) for value in np.percentile(data, [q * 100 for q in qs])]


def _discrete_order_statistic(counts: np.ndarray, k_1based: int) -> int:
    cumulative = np.cumsum(counts)
    return int(np.searchsorted(cumulative, k_1based, side="left"))


def discrete_median(counts: np.ndarray) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    if total % 2 == 1:
        return float(_discrete_order_statistic(counts, total // 2 + 1))
    left = _discrete_order_statistic(counts, total // 2)
    right = _discrete_order_statistic(counts, total // 2 + 1)
    return 0.5 * (left + right)


def discrete_quantile(counts: np.ndarray, q: float) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    k = int(np.ceil(q * total))
    k = min(max(k, 1), total)
    return float(_discrete_order_statistic(counts, k))


@dataclass
class AuditResult:
    variable_table: pd.DataFrame
    patients: pd.DataFrame
    cohort: dict
    trajectory: dict
    iculos_curves: pd.DataFrame
    correlations: pd.DataFrame
    unit_counts: pd.DataFrame
    read_errors: list[dict] = field(default_factory=list)


class CohortAuditor:
    def __init__(self) -> None:
        self.columns = list(EXPECTED_COLUMNS)
        self.n_vars = len(self.columns)
        self.index = {name: i for i, name in enumerate(self.columns)}
        self.label_index = self.index[TARGET]
        self.iculos_index = self.index["ICULOS"]
        self.fio2_index = self.index[FIO2_NAME]
        self.clinical_indexes = [self.index[name] for name in CLINICAL_MEASUREMENTS]
        self.input_indexes = [i for i, spec in enumerate(ALL_SPECS) if spec.role != "target"]
        n_h = len(HOSPITALS)
        self.n_hours = np.zeros(n_h, dtype=np.int64)
        self.n_patients = np.zeros(n_h, dtype=np.int64)
        self.n_sepsis_patients = np.zeros(n_h, dtype=np.int64)
        self.n_positive_hours = np.zeros(n_h, dtype=np.int64)
        self.n_positive_from_start = np.zeros(n_h, dtype=np.int64)
        self.n_nonmonotonic = np.zeros(n_h, dtype=np.int64)
        self.n_onset_inside = np.zeros(n_h, dtype=np.int64)
        self.n_iculos_not_step = np.zeros(n_h, dtype=np.int64)
        self.n_iculos_not_start1 = np.zeros(n_h, dtype=np.int64)
        self.observed = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.patients_ever = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.patients_ge2 = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.patients_changed = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.first_at_start = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.first_after_start = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.outside_bounds = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.miss_label0 = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.obs_label0 = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.miss_label1 = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.obs_label1 = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.gap_counts = np.zeros((n_h, self.n_vars, MAX_GAP_HOURS + 1), dtype=np.int64)
        self.gap_overflow = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.max_gap = np.zeros((n_h, self.n_vars), dtype=np.int64)
        self.hour_welford = [[_Welford() for _ in range(self.n_vars)] for _ in range(n_h)]
        self.patient_welford = [[_Welford() for _ in range(self.n_vars)] for _ in range(n_h)]
        streams = np.random.SeedSequence(RESERVOIR_SEED).spawn(self.n_vars)
        self.reservoirs = [_Reservoir(RESERVOIR_SIZE, np.random.default_rng(stream)) for stream in streams]
        self.fio2_le_1 = np.zeros(n_h, dtype=np.int64)
        self.fio2_gt_1 = np.zeros(n_h, dtype=np.int64)
        self.los_counts = np.zeros((n_h, MAX_GAP_HOURS + 1), dtype=np.int64)
        self.los_overflow = np.zeros(n_h, dtype=np.int64)
        n_off = TRAJECTORY_MAX_OFFSET - TRAJECTORY_MIN_OFFSET + 1
        self.traj_vars = list(TRAJECTORY_VARIABLES)
        self.traj_sum = np.zeros((n_h, len(self.traj_vars), n_off), dtype=np.float64)
        self.traj_n = np.zeros((n_h, len(self.traj_vars), n_off), dtype=np.int64)
        self.compare_vars = list(COMPARE_VARIABLES)
        self.curve_sum = np.zeros((n_h, 2, len(self.compare_vars), ICULOS_CURVE_MAX + 1), dtype=np.float64)
        self.curve_n = np.zeros((n_h, 2, len(self.compare_vars), ICULOS_CURVE_MAX + 1), dtype=np.int64)
        self.unit_counter: dict[tuple[str, str], int] = {}
        self.n_pair = np.zeros((self.n_vars, self.n_vars), dtype=np.int64)
        self.sum_x = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.sum_y = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.sum_xy = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.sum_x2 = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.sum_y2 = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.patient_rows: list[dict] = []
        self.read_errors: list[dict] = []
        self.seen_ids: set[str] = set()

    def add_patient(self, hospital: str, patient_id: str, values: np.ndarray) -> None:
        if hospital not in HOSPITALS:
            raise ValueError(hospital)
        if patient_id in self.seen_ids:
            raise ValueError(f"duplicate patient id {patient_id}")
        self.seen_ids.add(patient_id)
        h = 0 if hospital == "A" else 1
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.n_vars:
            raise ValueError(f"{patient_id}: expected shape (T, {self.n_vars})")
        n_rows = int(array.shape[0])
        self.n_patients[h] += 1
        self.n_hours[h] += n_rows
        if n_rows <= MAX_GAP_HOURS:
            self.los_counts[h, n_rows] += 1
        else:
            self.los_overflow[h] += 1

        labels = array[:, self.label_index]
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError(f"{patient_id}: SepsisLabel is not binary")
        positive = labels == 1
        n_positive = int(positive.sum())
        self.n_positive_hours[h] += n_positive
        sepsis = n_positive > 0
        if sepsis:
            self.n_sepsis_patients[h] += 1
        monotonic = label_is_monotonic_nondecreasing(labels)
        if not monotonic:
            self.n_nonmonotonic[h] += 1
        first = int(np.flatnonzero(positive)[0]) if sepsis else None
        positive_from_start = bool(sepsis and first == 0)
        if positive_from_start:
            self.n_positive_from_start[h] += 1
        onset = None if first is None else int(first - DT_OPTIMAL)
        if onset is not None and onset < n_rows:
            self.n_onset_inside[h] += 1

        iculos = array[:, self.iculos_index]
        if n_rows == 1:
            step_ok = True
        else:
            step_ok = bool(np.all(np.diff(iculos) == 1))
        start_ok = bool(iculos[0] == 1)
        if not step_ok:
            self.n_iculos_not_step[h] += 1
        if not start_ok:
            self.n_iculos_not_start1[h] += 1

        patient_obs = np.zeros(self.n_vars, dtype=np.int32)
        finite_cols: list[int] = []
        for j in range(self.n_vars):
            column = array[:, j]
            observed_idx = np.flatnonzero(~np.isnan(column))
            count = int(observed_idx.size)
            patient_obs[j] = count
            if count == 0:
                self.miss_label0[h, j] += int(np.sum(~positive))
                self.miss_label1[h, j] += n_positive
                continue
            finite_cols.append(j)
            vals = column[observed_idx]
            self.observed[h, j] += count
            self.patients_ever[h, j] += 1
            if observed_idx[0] == 0:
                self.first_at_start[h, j] += 1
            else:
                self.first_after_start[h, j] += 1
            if count >= 2:
                self.patients_ge2[h, j] += 1
                if float(vals.min()) != float(vals.max()):
                    self.patients_changed[h, j] += 1
                gaps = np.diff(observed_idx).astype(np.int64)
                self.max_gap[h, j] = max(int(self.max_gap[h, j]), int(gaps.max()))
                in_range = gaps[gaps <= MAX_GAP_HOURS]
                if in_range.size:
                    self.gap_counts[h, j] += np.bincount(in_range, minlength=MAX_GAP_HOURS + 1)
                self.gap_overflow[h, j] += int(np.sum(gaps > MAX_GAP_HOURS))
            self.hour_welford[h][j].add(vals)
            spec = ALL_SPECS[j]
            if spec.audit_low is not None and spec.audit_high is not None:
                self.outside_bounds[h, j] += int(np.sum((vals < spec.audit_low) | (vals > spec.audit_high)))
            if j == self.fio2_index:
                self.fio2_le_1[h] += int(np.sum(vals <= 1))
                self.fio2_gt_1[h] += int(np.sum(vals > 1))
            if hospital == "A" or hospital == "B":
                # Reservoir is pooled across hospitals. One seeded stream per variable.
                self.reservoirs[j].add(vals)
            missing = np.isnan(column)
            self.miss_label0[h, j] += int(np.sum(missing & ~positive))
            self.obs_label0[h, j] += int(np.sum(~missing & ~positive))
            self.miss_label1[h, j] += int(np.sum(missing & positive))
            self.obs_label1[h, j] += int(np.sum(~missing & positive))
            if spec.role == "static" and float(vals.min()) == float(vals.max()):
                self.patient_welford[h][j].add(np.array([float(vals[0])]))

        self._update_correlations(array, finite_cols)
        self._update_trajectory(h, array, positive, first)
        self._update_iculos_curves(h, array, positive, sepsis)
        unit_key = self._unit_key(array)
        self.unit_counter[(hospital, unit_key)] = self.unit_counter.get((hospital, unit_key), 0) + 1

        age = _static_value(array[:, self.index["Age"]])
        gender = _static_value(array[:, self.index["Gender"]])
        hosp_adm = _static_value(array[:, self.index["HospAdmTime"]])
        clinical_obs = int(patient_obs[self.clinical_indexes].sum())
        clinical_cells = n_rows * len(self.clinical_indexes)
        input_obs = int(patient_obs[self.input_indexes].sum())
        input_cells = n_rows * len(self.input_indexes)
        never = int(np.sum(patient_obs[self.clinical_indexes] == 0))
        self.patient_rows.append(
            {
                "patient_id": patient_id,
                "hospital_system": hospital,
                "n_hours": n_rows,
                "sepsis_patient": int(sepsis),
                "n_positive_hours": n_positive,
                "first_positive_hour": None if first is None else first + 1,
                "label_positive_from_first_hour": int(positive_from_start),
                "label_monotonic_nondecreasing": int(monotonic),
                "reconstructed_onset_hour": None if onset is None else onset + 1,
                "onset_inside_record": int(onset is not None and onset < n_rows),
                "iculos_start": float(iculos[0]),
                "iculos_end": float(iculos[-1]),
                "iculos_steps_by_one": int(step_ok),
                "age": age,
                "gender": gender,
                "hosp_adm_time": hosp_adm,
                "n_vital_lab_never_observed": never,
                "frac_vital_lab_cells_missing": 1.0 - clinical_obs / clinical_cells,
                "frac_input_cells_missing": 1.0 - input_obs / input_cells,
            }
        )

    def _update_correlations(self, array: np.ndarray, columns: list[int]) -> None:
        # Skip the target column. ICULOS is retained so its association with time can be seen.
        usable = [j for j in columns if j != self.label_index]
        for left_pos, left in enumerate(usable):
            x = array[:, left]
            x_ok = ~np.isnan(x)
            for right in usable[left_pos:]:
                y = array[:, right]
                mask = x_ok & ~np.isnan(y)
                if not np.any(mask):
                    continue
                xv = x[mask]
                yv = y[mask]
                self.n_pair[left, right] += xv.size
                self.sum_x[left, right] += float(xv.sum())
                self.sum_y[left, right] += float(yv.sum())
                self.sum_xy[left, right] += float(np.dot(xv, yv))
                self.sum_x2[left, right] += float(np.dot(xv, xv))
                self.sum_y2[left, right] += float(np.dot(yv, yv))

    def _update_trajectory(self, h: int, array: np.ndarray, positive: np.ndarray, first: int | None) -> None:
        if first is None or first == 0:
            return
        offsets = np.arange(array.shape[0]) - first
        keep = (offsets >= TRAJECTORY_MIN_OFFSET) & (offsets <= TRAJECTORY_MAX_OFFSET) & ~positive
        if not np.any(keep):
            return
        for var_i, name in enumerate(self.traj_vars):
            column = array[:, self.index[name]]
            for t in np.flatnonzero(keep):
                value = column[t]
                if np.isnan(value):
                    continue
                bin_index = int(offsets[t] - TRAJECTORY_MIN_OFFSET)
                self.traj_sum[h, var_i, bin_index] += float(value)
                self.traj_n[h, var_i, bin_index] += 1

    def _update_iculos_curves(self, h: int, array: np.ndarray, positive: np.ndarray, sepsis: bool) -> None:
        iculos = array[:, self.iculos_index]
        valid_time = (iculos >= 1) & (iculos <= ICULOS_CURVE_MAX) & (np.round(iculos) == iculos)
        if sepsis:
            valid_time = valid_time & ~positive
            group = 1
        else:
            group = 0
        hours = iculos[valid_time].astype(int)
        if hours.size == 0:
            return
        for var_i, name in enumerate(self.compare_vars):
            column = array[valid_time, self.index[name]]
            ok = ~np.isnan(column)
            if not np.any(ok):
                continue
            np.add.at(self.curve_sum[h, group, var_i], hours[ok], column[ok])
            np.add.at(self.curve_n[h, group, var_i], hours[ok], 1)

    def _unit_key(self, array: np.ndarray) -> str:
        unit1 = _static_token(array[:, self.index["Unit1"]])
        unit2 = _static_token(array[:, self.index["Unit2"]])
        return f"{unit1}|{unit2}"

    def finalize(self) -> AuditResult:
        variable_table = self._variable_table()
        patients = pd.DataFrame(self.patient_rows)
        cohort = self._cohort_dict(patients)
        trajectory = self._trajectory_dict()
        curves = self._curve_frame()
        correlations = self._correlation_frame()
        units = (
            pd.DataFrame(
                [{"hospital_system": key[0], "unit1_unit2": key[1], "n_patients": count} for key, count in sorted(self.unit_counter.items())]
            )
            if self.unit_counter
            else pd.DataFrame(columns=["hospital_system", "unit1_unit2", "n_patients"])
        )
        return AuditResult(
            variable_table=variable_table,
            patients=patients,
            cohort=cohort,
            trajectory=trajectory,
            iculos_curves=curves,
            correlations=correlations,
            unit_counts=units,
            read_errors=self.read_errors,
        )

    def _variable_table(self) -> pd.DataFrame:
        frames = []
        for h, hospital in enumerate(HOSPITALS):
            if int(self.n_patients[h]) == 0:
                continue
            frames.append(self._variable_rows(h, hospital))
        pooled = self._pooled_rows()
        table = pd.concat(frames + [pooled], ignore_index=True)
        return table

    def _variable_rows(self, h: int, hospital: str) -> pd.DataFrame:
        rows = []
        n_hours = int(self.n_hours[h])
        n_patients = int(self.n_patients[h])
        for j, spec in enumerate(ALL_SPECS):
            gaps = self.gap_counts[h, j]
            median_gap = discrete_median(gaps)
            p90_gap = discrete_quantile(gaps, 0.90)
            pct_hours = 100.0 * float(self.observed[h, j]) / n_hours if n_hours else float("nan")
            ever = int(self.patients_ever[h, j])
            ge2 = int(self.patients_ge2[h, j])
            changed = int(self.patients_changed[h, j])
            stats = self.hour_welford[h][j]
            rows.append(
                {
                    "variable": spec.name,
                    "hospital_system": hospital,
                    "role": spec.role,
                    "a_priori_group": _a_priori(spec.name, spec.role),
                    "unit": spec.unit,
                    "description": spec.description,
                    "n_hours": n_hours,
                    "n_patients": n_patients,
                    "n_observed": int(self.observed[h, j]),
                    "pct_hours_observed": pct_hours,
                    "pct_hours_missing": 100.0 - pct_hours if n_hours else float("nan"),
                    "n_patients_ever_observed": ever,
                    "pct_patients_ever_observed": 100.0 * ever / n_patients if n_patients else float("nan"),
                    "n_patients_never_observed": n_patients - ever,
                    "n_patients_at_least_two_observations": ge2,
                    "n_patients_with_value_change": changed,
                    "pct_repeat_patients_with_change": 100.0 * changed / ge2 if ge2 else float("nan"),
                    "n_gaps": int(gaps.sum()),
                    "median_gap_hours": median_gap,
                    "p90_gap_hours": p90_gap,
                    "max_gap_hours": int(self.max_gap[h, j]) if ge2 else None,
                    "gap_overflow_count": int(self.gap_overflow[h, j]),
                    "mean": None if stats.n == 0 else stats.mean,
                    "std": None if stats.n < 2 else stats.std,
                    "min": None if stats.n == 0 else stats.minimum,
                    "max": None if stats.n == 0 else stats.maximum,
                    "n_outside_audit_bounds": int(self.outside_bounds[h, j]),
                    "audit_low": spec.audit_low,
                    "audit_high": spec.audit_high,
                    "n_patients_first_observation_at_hour_1": int(self.first_at_start[h, j]),
                    "n_patients_first_observation_after_hour_1": int(self.first_after_start[h, j]),
                    "n_missing_label0": int(self.miss_label0[h, j]),
                    "n_observed_label0": int(self.obs_label0[h, j]),
                    "n_missing_label1": int(self.miss_label1[h, j]),
                    "n_observed_label1": int(self.obs_label1[h, j]),
                }
            )
        return pd.DataFrame(rows)

    def _pooled_rows(self) -> pd.DataFrame:
        rows = []
        n_hours = int(self.n_hours.sum())
        n_patients = int(self.n_patients.sum())
        for j, spec in enumerate(ALL_SPECS):
            gaps = self.gap_counts[0, j] + self.gap_counts[1, j]
            stats = _combine_welford(self.hour_welford[0][j], self.hour_welford[1][j])
            ever = int(self.patients_ever[:, j].sum())
            ge2 = int(self.patients_ge2[:, j].sum())
            changed = int(self.patients_changed[:, j].sum())
            pct_hours = 100.0 * float(self.observed[:, j].sum()) / n_hours if n_hours else float("nan")
            median_gap = discrete_median(gaps)
            empirical = empirical_frequency_class(spec.role, pct_hours, median_gap)
            change_pct = 100.0 * changed / ge2 if ge2 else 0.0
            p05, p50, p95 = self.reservoirs[j].percentiles((0.05, 0.50, 0.95))
            max_gap_candidates = [int(self.max_gap[h, j]) for h in range(2) if self.patients_ge2[h, j] > 0]
            rows.append(
                {
                    "variable": spec.name,
                    "hospital_system": "pooled",
                    "role": spec.role,
                    "a_priori_group": _a_priori(spec.name, spec.role),
                    "unit": spec.unit,
                    "description": spec.description,
                    "n_hours": n_hours,
                    "n_patients": n_patients,
                    "n_observed": int(self.observed[:, j].sum()),
                    "pct_hours_observed": pct_hours,
                    "pct_hours_missing": 100.0 - pct_hours if n_hours else float("nan"),
                    "n_patients_ever_observed": ever,
                    "pct_patients_ever_observed": 100.0 * ever / n_patients if n_patients else float("nan"),
                    "n_patients_never_observed": n_patients - ever,
                    "n_patients_at_least_two_observations": ge2,
                    "n_patients_with_value_change": changed,
                    "pct_repeat_patients_with_change": 100.0 * changed / ge2 if ge2 else float("nan"),
                    "n_gaps": int(gaps.sum()),
                    "median_gap_hours": median_gap,
                    "p90_gap_hours": discrete_quantile(gaps, 0.90),
                    "max_gap_hours": None if not max_gap_candidates else max(max_gap_candidates),
                    "gap_overflow_count": int(self.gap_overflow[:, j].sum()),
                    "mean": None if stats.n == 0 else stats.mean,
                    "std": None if stats.n < 2 else stats.std,
                    "min": None if stats.n == 0 else stats.minimum,
                    "max": None if stats.n == 0 else stats.maximum,
                    "reservoir_p05": p05,
                    "reservoir_p50": p50,
                    "reservoir_p95": p95,
                    "reservoir_n": int(min(self.reservoirs[j].seen, self.reservoirs[j].size)),
                    "reservoir_population_n": int(self.reservoirs[j].seen),
                    "n_outside_audit_bounds": int(self.outside_bounds[:, j].sum()),
                    "audit_low": spec.audit_low,
                    "audit_high": spec.audit_high,
                    "n_patients_first_observation_at_hour_1": int(self.first_at_start[:, j].sum()),
                    "n_patients_first_observation_after_hour_1": int(self.first_after_start[:, j].sum()),
                    "n_missing_label0": int(self.miss_label0[:, j].sum()),
                    "n_observed_label0": int(self.obs_label0[:, j].sum()),
                    "n_missing_label1": int(self.miss_label1[:, j].sum()),
                    "n_observed_label1": int(self.obs_label1[:, j].sum()),
                    "empirical_class": empirical,
                    "a_priori_empirical_disagreement": _group_disagreement(spec.name, spec.role, empirical),
                    "representation_recommendation": recommend_representation(spec.role, empirical, change_pct),
                    "candidate_features": ";".join(candidate_feature_names(empirical, spec.name)),
                }
            )
        pooled = pd.DataFrame(rows)
        class_a = self._class_map(0)
        class_b = self._class_map(1)
        pooled["empirical_class_set_A"] = pooled["variable"].map(class_a)
        pooled["empirical_class_set_B"] = pooled["variable"].map(class_b)
        pooled["hospital_class_disagreement"] = pooled["empirical_class_set_A"] != pooled["empirical_class_set_B"]
        return pooled

    def _class_map(self, h: int) -> dict[str, str]:
        n_hours = int(self.n_hours[h])
        mapping = {}
        for j, spec in enumerate(ALL_SPECS):
            pct = 100.0 * float(self.observed[h, j]) / n_hours if n_hours else float("nan")
            median_gap = discrete_median(self.gap_counts[h, j])
            mapping[spec.name] = empirical_frequency_class(spec.role, pct, median_gap)
        return mapping

    def _cohort_dict(self, patients: pd.DataFrame) -> dict:
        def block(mask: np.ndarray, hospital: str) -> dict:
            n_patients = int(mask.sum()) if hospital != "pooled" else int(self.n_patients.sum())
            if hospital == "pooled":
                h_slice = slice(None)
                n_patients = int(self.n_patients.sum())
                n_hours = int(self.n_hours.sum())
                n_sepsis = int(self.n_sepsis_patients.sum())
                n_pos_hours = int(self.n_positive_hours.sum())
                n_from_start = int(self.n_positive_from_start.sum())
                n_nonmono = int(self.n_nonmonotonic.sum())
                n_onset = int(self.n_onset_inside.sum())
                n_step = int(self.n_iculos_not_step.sum())
                n_start = int(self.n_iculos_not_start1.sum())
                los = self.los_counts.sum(axis=0)
            else:
                h = 0 if hospital == "A" else 1
                n_patients = int(self.n_patients[h])
                n_hours = int(self.n_hours[h])
                n_sepsis = int(self.n_sepsis_patients[h])
                n_pos_hours = int(self.n_positive_hours[h])
                n_from_start = int(self.n_positive_from_start[h])
                n_nonmono = int(self.n_nonmonotonic[h])
                n_onset = int(self.n_onset_inside[h])
                n_step = int(self.n_iculos_not_step[h])
                n_start = int(self.n_iculos_not_start1[h])
                los = self.los_counts[h]
            part = patients[patients["hospital_system"] == hospital] if hospital != "pooled" else patients
            return {
                "hospital_system": hospital,
                "n_patients": n_patients,
                "n_hours": n_hours,
                "n_sepsis_patients": n_sepsis,
                "n_nonsepsis_patients": n_patients - n_sepsis,
                "patient_sepsis_prevalence": n_sepsis / n_patients if n_patients else None,
                "n_positive_hours": n_pos_hours,
                "n_negative_hours": n_hours - n_pos_hours,
                "hour_positive_prevalence": n_pos_hours / n_hours if n_hours else None,
                "n_sepsis_patients_positive_from_first_hour": n_from_start,
                "n_sepsis_patients_with_observed_prewindow": n_sepsis - n_from_start,
                "n_sepsis_patients_onset_inside_record": n_onset,
                "n_patients_label_not_monotonic": n_nonmono,
                "n_patients_iculos_not_incrementing_by_1": n_step,
                "n_patients_iculos_not_starting_at_1": n_start,
                "hours_per_patient": _los_summary(los, int(self.los_overflow.sum() if hospital == "pooled" else self.los_overflow[0 if hospital == "A" else 1])),
                "patient_median_frac_vital_lab_cells_missing": None
                if part.empty
                else float(part["frac_vital_lab_cells_missing"].median()),
                "patient_mean_n_vital_lab_never_observed": None
                if part.empty
                else float(part["n_vital_lab_never_observed"].mean()),
            }

        fio2 = {
            hospital: {
                "n_values_le_1": int(self.fio2_le_1[h]),
                "n_values_gt_1": int(self.fio2_gt_1[h]),
            }
            for h, hospital in enumerate(HOSPITALS)
        }
        fio2["pooled"] = {
            "n_values_le_1": int(self.fio2_le_1.sum()),
            "n_values_gt_1": int(self.fio2_gt_1.sum()),
        }
        static_names = [spec.name for spec in ALL_SPECS if spec.role == "static"]
        patient_static = {}
        for h, hospital in enumerate(HOSPITALS):
            patient_static[hospital] = {}
            for name in static_names:
                stats = self.patient_welford[h][self.index[name]]
                patient_static[hospital][name] = {
                    "n_patients_with_constant_value": stats.n,
                    "mean": None if stats.n == 0 else stats.mean,
                    "std": None if stats.n < 2 else stats.std,
                    "min": None if stats.n == 0 else stats.minimum,
                    "max": None if stats.n == 0 else stats.maximum,
                }
        pooled_static = {}
        for name in static_names:
            stats = _combine_welford(
                self.patient_welford[0][self.index[name]],
                self.patient_welford[1][self.index[name]],
            )
            pooled_static[name] = {
                "n_patients_with_constant_value": stats.n,
                "mean": None if stats.n == 0 else stats.mean,
                "std": None if stats.n < 2 else stats.std,
                "min": None if stats.n == 0 else stats.minimum,
                "max": None if stats.n == 0 else stats.maximum,
            }
        patient_static["pooled"] = pooled_static
        return {
            "hospitals": {name: block(np.array([]), name) for name in ("A", "B", "pooled")},
            "patient_level_static": patient_static,
            "fio2_scale_counts": fio2,
            "trajectory_alignment": (
                "Offsets are hours relative to the first SepsisLabel=1 row. "
                "That row is t_sepsis - 6 in the challenge definition, not the onset hour. "
                "Patients already positive at hour 1 are excluded because no pre-label hour was recorded."
            ),
            "reservoir": {
                "size": RESERVOIR_SIZE,
                "seed": RESERVOIR_SEED,
                "note": "Percentiles are reservoir estimates. Counts, means, standard deviations, minima, maxima, and missingness are exact.",
            },
            "gap_definition": (
                "A gap is the difference in row index between consecutive non-missing values. "
                "When ICULOS increments by 1 each row, that difference is hours between real measurements."
            ),
        }

    def _trajectory_dict(self) -> dict:
        offsets = list(range(TRAJECTORY_MIN_OFFSET, TRAJECTORY_MAX_OFFSET + 1))
        hospitals = {}
        arrays = {"pooled": self.traj_sum.sum(axis=0)}
        counts = {"pooled": self.traj_n.sum(axis=0)}
        for h, name in enumerate(HOSPITALS):
            arrays[name] = self.traj_sum[h]
            counts[name] = self.traj_n[h]
        for name in ("A", "B", "pooled"):
            hospitals[name] = {}
            for var_i, variable in enumerate(self.traj_vars):
                means = []
                ns = []
                for bin_index, _offset in enumerate(offsets):
                    n = int(counts[name][var_i, bin_index])
                    ns.append(n)
                    means.append(None if n == 0 else float(arrays[name][var_i, bin_index] / n))
                hospitals[name][variable] = {"n": ns, "mean": means}
        return {"offsets_hours_relative_to_first_positive_label": offsets, "hospitals": hospitals}

    def _curve_frame(self) -> pd.DataFrame:
        rows = []
        for h, hospital in enumerate(HOSPITALS):
            for hour in range(1, ICULOS_CURVE_MAX + 1):
                for var_i, variable in enumerate(self.compare_vars):
                    for group, group_name in ((0, "nonsepsis"), (1, "sepsis_prelabel")):
                        n = int(self.curve_n[h, group, var_i, hour])
                        if n == 0:
                            continue
                        rows.append(
                            {
                                "hospital_system": hospital,
                                "iculos_hour": hour,
                                "variable": variable,
                                "group": group_name,
                                "n": n,
                                "mean": float(self.curve_sum[h, group, var_i, hour] / n),
                            }
                        )
        return pd.DataFrame(rows)

    def _correlation_frame(self) -> pd.DataFrame:
        rows = []
        for left in range(self.n_vars):
            for right in range(left + 1, self.n_vars):
                if self.columns[left] == TARGET or self.columns[right] == TARGET:
                    continue
                n = int(self.n_pair[left, right])
                if n < 3:
                    continue
                sx = self.sum_x[left, right]
                sy = self.sum_y[left, right]
                sxy = self.sum_xy[left, right]
                sx2 = self.sum_x2[left, right]
                sy2 = self.sum_y2[left, right]
                cov = sxy - sx * sy / n
                var_x = sx2 - sx * sx / n
                var_y = sy2 - sy * sy / n
                if var_x <= 0 or var_y <= 0:
                    continue
                corr = cov / np.sqrt(var_x * var_y)
                rows.append(
                    {
                        "variable_a": self.columns[left],
                        "variable_b": self.columns[right],
                        "n_paired_hours": n,
                        "pearson_r": float(corr),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["variable_a", "variable_b", "n_paired_hours", "pearson_r"])
        frame = pd.DataFrame(rows)
        return frame.reindex(frame["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _combine_welford(left: _Welford, right: _Welford) -> _Welford:
    combined = _Welford()
    if left.n == 0:
        return right
    if right.n == 0:
        return left
    combined.n = left.n + right.n
    delta = right.mean - left.mean
    combined.mean = left.mean + delta * right.n / combined.n
    combined.m2 = left.m2 + right.m2 + delta * delta * left.n * right.n / combined.n
    combined.minimum = min(left.minimum, right.minimum)
    combined.maximum = max(left.maximum, right.maximum)
    return combined


def _static_value(column: np.ndarray) -> float | None:
    observed = column[~np.isnan(column)]
    if observed.size == 0:
        return None
    if float(observed.min()) != float(observed.max()):
        return None
    return float(observed[0])


def _static_token(column: np.ndarray) -> str:
    observed = column[~np.isnan(column)]
    if observed.size == 0:
        return "missing"
    if float(observed.min()) != float(observed.max()):
        return "inconsistent"
    value = float(observed[0])
    if value.is_integer():
        return str(int(value))
    return str(value)


def _a_priori(name: str, role: str) -> str:
    if name in GROUP_A_RAPID:
        return "A_rapid_vital"
    if name in GROUP_B_SPARSE:
        return "B_sparse_lab"
    if name in GROUP_C_STATIC:
        return "C_static"
    if role == "time_context":
        return "time_context"
    if role == "target":
        return "target"
    return role


def _group_disagreement(name: str, role: str, empirical: str) -> str:
    if role in {"static", "time_context", "target"}:
        return ""
    if name in GROUP_A_RAPID and empirical == "sparse":
        return "a priori rapid vital is empirically sparse"
    if name in GROUP_A_RAPID and empirical == "intermittent":
        return "a priori rapid vital is only intermittent"
    if name in GROUP_B_SPARSE and empirical == "near_hourly":
        return "a priori sparse lab is empirically near-hourly"
    return ""


def _los_summary(counts: np.ndarray, overflow: int) -> dict:
    total = int(counts.sum())
    return {
        "n_patients_in_histogram": total,
        "n_patients_above_histogram": int(overflow),
        "min": None if total == 0 else discrete_quantile(counts, 0),
        "p25": discrete_quantile(counts, 0.25),
        "median": discrete_median(counts),
        "p75": discrete_quantile(counts, 0.75),
        "p95": discrete_quantile(counts, 0.95),
        "max_in_histogram": None if total == 0 else int(np.flatnonzero(counts).max()),
        "mean": None if total == 0 else float(np.sum(np.arange(counts.size) * counts) / total),
    }


def iter_patient_files(raw_root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for hospital, folder in (("A", "training_setA"), ("B", "training_setB")):
        directory = raw_root / folder
        if not directory.exists():
            continue
        files.extend((hospital, path) for path in sorted(directory.glob("p*.psv")))
    return files


def audit_directory(raw_root: Path) -> AuditResult:
    auditor = CohortAuditor()
    files = iter_patient_files(raw_root)
    if not files:
        raise FileNotFoundError(f"No patient files under {raw_root}")
    started = time.perf_counter()
    for number, (hospital, path) in enumerate(files, start=1):
        try:
            header, values = read_psv(path)
            validate_header(header, path)
            auditor.add_patient(hospital, path.stem, values)
        except Exception as exc:  # recorded; a header failure still aborts below if it is systemic
            auditor.read_errors.append({"patient_id": path.stem, "hospital_system": hospital, "error": str(exc)})
            if "header does not match" in str(exc):
                raise
        if number % 2000 == 0 or number == len(files):
            elapsed = time.perf_counter() - started
            print(f"audited {number}/{len(files)} files in {elapsed:.1f}s; errors={len(auditor.read_errors)}", flush=True)
    return auditor.finalize()
