export interface ParameterField {
  name: string;
  label: string;
  unit: string;
  kind?: "binary";
  options?: Array<{ value: string; label: string }>;
}

export interface ParameterGroup {
  id: string;
  title: string;
  fields: ParameterField[];
}

const binaryOptions = [
  { value: "0", label: "0" },
  { value: "1", label: "1" },
];

export const PARAMETER_GROUPS: ParameterGroup[] = [
  {
    id: "vitals",
    title: "Vital signs",
    fields: [
      { name: "HR", label: "Heart rate", unit: "beats per minute" },
      { name: "O2Sat", label: "Oxygen saturation", unit: "%" },
      { name: "Temp", label: "Temperature", unit: "°C" },
      { name: "SBP", label: "Systolic blood pressure", unit: "mm Hg" },
      { name: "MAP", label: "Mean arterial pressure", unit: "mm Hg" },
      { name: "DBP", label: "Diastolic blood pressure", unit: "mm Hg" },
      { name: "Resp", label: "Respiratory rate", unit: "breaths per minute" },
      { name: "EtCO2", label: "End-tidal CO2", unit: "mm Hg" },
    ],
  },
  {
    id: "labs",
    title: "Laboratory",
    fields: [
      { name: "BaseExcess", label: "Base excess", unit: "mmol/L" },
      { name: "HCO3", label: "HCO3", unit: "mmol/L" },
      { name: "FiO2", label: "FiO2", unit: "as recorded" },
      { name: "pH", label: "pH", unit: "pH" },
      { name: "PaCO2", label: "PaCO2", unit: "mm Hg" },
      { name: "SaO2", label: "SaO2", unit: "%" },
      { name: "AST", label: "AST", unit: "IU/L" },
      { name: "BUN", label: "BUN", unit: "mg/dL" },
      { name: "Alkalinephos", label: "Alkaline phosphatase", unit: "IU/L" },
      { name: "Calcium", label: "Calcium", unit: "mg/dL" },
      { name: "Chloride", label: "Chloride", unit: "mmol/L" },
      { name: "Creatinine", label: "Creatinine", unit: "mg/dL" },
      { name: "Bilirubin_direct", label: "Direct bilirubin", unit: "mg/dL" },
      { name: "Glucose", label: "Glucose", unit: "mg/dL" },
      { name: "Lactate", label: "Lactate", unit: "as recorded" },
      { name: "Magnesium", label: "Magnesium", unit: "as recorded" },
      { name: "Phosphate", label: "Phosphate", unit: "mg/dL" },
      { name: "Potassium", label: "Potassium", unit: "mmol/L" },
      { name: "Bilirubin_total", label: "Total bilirubin", unit: "mg/dL" },
      { name: "TroponinI", label: "Troponin I", unit: "ng/mL" },
      { name: "Hct", label: "Hematocrit", unit: "%" },
      { name: "Hgb", label: "Hemoglobin", unit: "g/dL" },
      { name: "PTT", label: "PTT", unit: "seconds" },
      { name: "WBC", label: "WBC", unit: "count ×10³/µL" },
      { name: "Fibrinogen", label: "Fibrinogen", unit: "mg/dL" },
      { name: "Platelets", label: "Platelets", unit: "count ×10³/µL" },
    ],
  },
  {
    id: "context",
    title: "Patient and context",
    fields: [
      { name: "Age", label: "Age", unit: "years" },
      {
        name: "Gender",
        label: "Gender",
        unit: "0 female, 1 male",
        kind: "binary",
        options: [
          { value: "0", label: "0 — female" },
          { value: "1", label: "1 — male" },
        ],
      },
      { name: "Unit1", label: "Unit1", unit: "administrative code", kind: "binary", options: binaryOptions },
      { name: "Unit2", label: "Unit2", unit: "administrative code", kind: "binary", options: binaryOptions },
      { name: "HospAdmTime", label: "Hospital admission time", unit: "hours before ICU admit" },
      { name: "ICULOS", label: "ICU length of stay", unit: "hours" },
    ],
  },
];

export const PARAMETER_LABELS: Record<string, string> = Object.fromEntries(
  PARAMETER_GROUPS.flatMap((group) => group.fields.map((field) => [field.name, field.label])),
);

export function emptyFields(): Record<string, string> {
  return Object.fromEntries(PARAMETER_GROUPS.flatMap((group) => group.fields.map((field) => [field.name, ""])));
}
