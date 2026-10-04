"""Official PhysioNet/CinC 2019 column schema and a priori clinical groups.

Units and descriptions follow the challenge data dictionary as published at
https://physionet.org/content/challenge-2019/1.0.0/
They are not silently converted when the published unit differs from a more
common clinical convention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: str  # vital, lab, static, time_context, target
    unit: str
    description: str
    # Broad screening bounds for counting implausible values. Not deletion rules.
    audit_low: float | None = None
    audit_high: float | None = None


VITAL_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec("HR", "vital", "beats per minute", "Heart rate", 0, 300),
    ColumnSpec("O2Sat", "vital", "%", "Pulse oximetry", 0, 100),
    ColumnSpec("Temp", "vital", "Deg C", "Temperature", 25, 45),
    ColumnSpec("SBP", "vital", "mm Hg", "Systolic blood pressure", 0, 300),
    ColumnSpec("MAP", "vital", "mm Hg", "Mean arterial pressure", 0, 300),
    ColumnSpec("DBP", "vital", "mm Hg", "Diastolic blood pressure", 0, 300),
    ColumnSpec("Resp", "vital", "breaths per minute", "Respiration rate", 0, 100),
    ColumnSpec("EtCO2", "vital", "mm Hg", "End-tidal carbon dioxide", 0, 100),
)

# Official table lists FiO2 with laboratory values even though it is often a
# ventilator setting. The a priori group follows that published table.
LAB_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec("BaseExcess", "lab", "mmol/L", "Measure of excess bicarbonate", -30, 30),
    ColumnSpec("HCO3", "lab", "mmol/L", "Bicarbonate", 0, 60),
    ColumnSpec("FiO2", "lab", "%", "Fraction of inspired oxygen", 0.0, 100.0),
    ColumnSpec("pH", "lab", "not stated", "pH", 6.5, 8.0),
    ColumnSpec("PaCO2", "lab", "mm Hg", "Partial pressure of carbon dioxide from arterial blood", 0, 150),
    ColumnSpec("SaO2", "lab", "%", "Oxygen saturation from arterial blood", 0, 100),
    ColumnSpec("AST", "lab", "IU/L", "Aspartate transaminase", 0, 10000),
    ColumnSpec("BUN", "lab", "mg/dL", "Blood urea nitrogen", 0, 200),
    ColumnSpec("Alkalinephos", "lab", "IU/L", "Alkaline phosphatase", 0, 2000),
    ColumnSpec("Calcium", "lab", "mg/dL", "Calcium", 0, 20),
    ColumnSpec("Chloride", "lab", "mmol/L", "Chloride", 50, 150),
    ColumnSpec("Creatinine", "lab", "mg/dL", "Creatinine", 0, 30),
    ColumnSpec("Bilirubin_direct", "lab", "mg/dL", "Bilirubin direct", 0, 50),
    ColumnSpec("Glucose", "lab", "mg/dL", "Serum glucose", 0, 2000),
    ColumnSpec("Lactate", "lab", "mg/dL", "Lactic acid (unit as published by the challenge)", 0, 30),
    ColumnSpec("Magnesium", "lab", "mmol/dL", "Magnesium (unit as published by the challenge)", 0, 10),
    ColumnSpec("Phosphate", "lab", "mg/dL", "Phosphate", 0, 20),
    ColumnSpec("Potassium", "lab", "mmol/L", "Potassium", 0, 12),
    ColumnSpec("Bilirubin_total", "lab", "mg/dL", "Total bilirubin", 0, 50),
    ColumnSpec("TroponinI", "lab", "ng/mL", "Troponin I", 0, 100),
    ColumnSpec("Hct", "lab", "%", "Hematocrit", 0, 80),
    ColumnSpec("Hgb", "lab", "g/dL", "Hemoglobin", 0, 25),
    ColumnSpec("PTT", "lab", "seconds", "Partial thromboplastin time", 0, 200),
    ColumnSpec("WBC", "lab", "count*10^3/uL", "Leukocyte count", 0, 200),
    ColumnSpec("Fibrinogen", "lab", "mg/dL", "Fibrinogen", 0, 1500),
    ColumnSpec("Platelets", "lab", "count*10^3/uL", "Platelets", 0, 2000),
)

STATIC_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec("Age", "static", "years", "Age in years; 100 means 90 or above", 0, 120),
    ColumnSpec("Gender", "static", "0 female, 1 male", "Sex coded as female (0) or male (1)", 0, 1),
    ColumnSpec("Unit1", "static", "binary administrative", "Administrative ICU unit identifier (MICU)", 0, 1),
    ColumnSpec("Unit2", "static", "binary administrative", "Administrative ICU unit identifier (SICU)", 0, 1),
    ColumnSpec(
        "HospAdmTime",
        "static",
        "hours",
        "Hours between hospital admit and ICU admit",
        -24 * 365,
        24 * 365,
    ),
)

TIME_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec("ICULOS", "time_context", "hours", "ICU length of stay; hours since ICU admit", 1, 10000),
)

TARGET_SPECS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "SepsisLabel",
        "target",
        "binary",
        "Challenge label: 1 from t_sepsis - 6 hours onward for sepsis patients; 0 otherwise. Not a contemporaneous confirmed diagnosis.",
        0,
        1,
    ),
)

ALL_SPECS: tuple[ColumnSpec, ...] = VITAL_SPECS + LAB_SPECS + STATIC_SPECS + TIME_SPECS + TARGET_SPECS
EXPECTED_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in ALL_SPECS)
COLUMN_BY_NAME: dict[str, ColumnSpec] = {spec.name: spec for spec in ALL_SPECS}

VITALS: tuple[str, ...] = tuple(spec.name for spec in VITAL_SPECS)
LABS: tuple[str, ...] = tuple(spec.name for spec in LAB_SPECS)
STATIC: tuple[str, ...] = tuple(spec.name for spec in STATIC_SPECS)
TIME_CONTEXT: tuple[str, ...] = tuple(spec.name for spec in TIME_SPECS)
TARGET = "SepsisLabel"
CLINICAL_MEASUREMENTS: tuple[str, ...] = VITALS + LABS

# A priori groups from the project brief. Empirical frequency may disagree.
GROUP_A_RAPID = VITALS
GROUP_B_SPARSE = LABS
GROUP_C_STATIC = STATIC

TRAJECTORY_VARIABLES: tuple[str, ...] = (
    "HR",
    "MAP",
    "Resp",
    "O2Sat",
    "Temp",
    "Lactate",
    "Creatinine",
    "WBC",
    "Platelets",
    "BUN",
)

# FiO2 is published both as a fraction and, in some files, possibly as a percent.
# The audit bound uses the 0-1 fraction scale AND a separate counter for values
# that look like percents (1 < x <= 100) so the scale can be inspected rather than assumed.
FIO2_NAME = "FiO2"

DATASET_DOI = "10.13026/v64v-d857"
DATASET_URL = "https://physionet.org/content/challenge-2019/1.0.0/"
TRAINING_FILES_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"
DOCUMENTED_SET_A_COUNT = 20336
DOCUMENTED_SET_B_COUNT = 20000
SPLIT_SEED = 2019
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
