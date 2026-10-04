export type DataSufficiency = "VERY_LOW" | "LOW" | "MODERATE" | "HIGH";

export type RiskTrend = "increasing" | "decreasing" | "unchanged" | null;

export type ExplanationDirection = "increases_risk_score" | "decreases_risk_score";

export interface PredictionHorizon {
  name: string;
  description: string;
}

export interface Explanation {
  feature: string;
  contribution: number;
  direction: ExplanationDirection;
  contribution_space: string;
}

export interface CurrentPrediction {
  risk_probability: number;
  research_early_warning: boolean;
  data_available: number;
  data_total: number;
  data_completeness: number;
  data_sufficiency: DataSufficiency;
  data_sufficiency_rule: string;
  historical_hours: number;
  observed_dynamic_variables: number;
  observed_laboratory_variables: number;
  has_prior_measurements: boolean;
  risk_trend: RiskTrend;
  prediction_horizon: PredictionHorizon;
  model_version: string;
  model_name: string;
  explanations: Explanation[];
  disclaimer: string;
}

export interface SequenceHourPrediction {
  hour: number | null;
  risk_probability: number;
  research_early_warning: boolean;
  data_available: number;
  data_total: number;
  data_completeness: number;
  data_sufficiency: DataSufficiency;
  data_sufficiency_rule: string;
  historical_hours: number;
  observed_dynamic_variables: number;
  observed_laboratory_variables: number;
  has_prior_measurements: boolean;
  risk_trend: RiskTrend;
  explanations: Explanation[];
}

export interface SequencePrediction {
  model_name: string;
  model_version: string;
  prediction_horizon: PredictionHorizon;
  disclaimer: string;
  predictions: SequenceHourPrediction[];
}

export interface ModelInfo {
  status: string;
  model_name: string;
  model_version: string;
  model_family: string;
  prediction_target: string;
  prediction_horizon: PredictionHorizon;
  supported_parameters: string[];
  flexible_input: boolean;
  calibration: string;
  decision_threshold: number;
  threshold_label: string;
  training_dataset_version: string;
  feature_schema_version: string;
  data_sufficiency_rule: string;
  data_sufficiency_description: string;
  disclaimer: string;
}

export interface ObservationRequest {
  observations: Array<{
    values: Record<string, number>;
  }>;
}

export interface FieldValues {
  [parameter: string]: string;
}
