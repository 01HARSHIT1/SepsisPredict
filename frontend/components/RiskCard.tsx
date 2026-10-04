import { formatCompleteness, formatRiskPercent, formatSufficiency, formatThreshold } from "@/lib/format";
import type { CurrentPrediction, ModelInfo } from "@/lib/types";

export function RiskCard({
  prediction,
  modelInfo,
  caption,
}: {
  prediction: CurrentPrediction | null;
  modelInfo: ModelInfo | null;
  caption?: string;
}) {
  if (!prediction) {
    return (
      <section className="rounded-lg border border-dashed border-line bg-panel p-5" aria-label="Estimated sepsis risk">
        <h2 className="text-lg font-semibold">Estimated Sepsis Risk</h2>
        <p className="mt-2 text-sm text-stone-700">Submit entered values to request a research early-warning estimate.</p>
      </section>
    );
  }

  const percent = formatRiskPercent(prediction.risk_probability);
  const thresholdText = prediction.research_early_warning
    ? "At or above the research early-warning threshold. Higher estimated model risk. This model output is not a diagnosis."
    : "Below the research early-warning threshold. Lower estimated model risk. This model output is not a diagnosis.";

  return (
    <section className="rounded-lg border border-line bg-panel p-5" aria-label="Estimated sepsis risk" aria-live="polite">
      <p className="text-xs font-semibold uppercase tracking-wide text-tideink">Research early-warning estimate</p>
      <h2 className="mt-1 text-lg font-semibold">Estimated Sepsis Risk</h2>
      {caption ? <p className="text-sm text-stone-700">{caption}</p> : null}
      <p className="mt-3 text-4xl font-semibold tabular-nums" data-testid="risk-percent">
        {percent}
      </p>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-stone-200" aria-hidden="true">
        <div className="h-full bg-tide" style={{ width: `${prediction.risk_probability * 100}%` }} />
      </div>
      <p className="mt-3 text-sm">{thresholdText}</p>
      {modelInfo ? (
        <p className="mt-1 text-sm text-stone-700">
          Research early-warning threshold: {formatThreshold(modelInfo.decision_threshold)}. This is not a diagnosis threshold.
        </p>
      ) : null}
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-stone-600">Data available</dt>
          <dd className="font-medium">
            {prediction.data_available} / {prediction.data_total}
          </dd>
        </div>
        <div>
          <dt className="text-stone-600">Data completeness</dt>
          <dd className="font-medium">{formatCompleteness(prediction.data_completeness)}</dd>
        </div>
        <div>
          <dt className="text-stone-600">Data sufficiency</dt>
          <dd className="font-medium" data-testid="risk-sufficiency">
            {formatSufficiency(prediction.data_sufficiency)}
          </dd>
        </div>
        <div>
          <dt className="text-stone-600">Historical hours</dt>
          <dd className="font-medium">{prediction.historical_hours}</dd>
        </div>
      </dl>
      <p className="mt-4 text-sm">Risk estimate is based only on the information submitted.</p>
      <p className="mt-2 text-sm text-stone-700">
        Prediction horizon: {prediction.prediction_horizon.description}
      </p>
      <p className="mt-2 break-all text-sm text-stone-700">
        Model: {prediction.model_name} · {prediction.model_version}
      </p>
    </section>
  );
}
