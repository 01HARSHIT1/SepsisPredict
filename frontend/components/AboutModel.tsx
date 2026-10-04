import { formatThreshold } from "@/lib/format";
import type { ModelInfo } from "@/lib/types";

export function AboutModel({ modelInfo, error }: { modelInfo: ModelInfo | null; error: string | null }) {
  return (
    <section className="rounded-lg border border-line bg-panel p-5" aria-label="About model">
      <h2 className="text-lg font-semibold">About model</h2>
      {error ? <p className="mt-2 text-sm">{error}</p> : null}
      {!modelInfo ? (
        <p className="mt-2 text-sm text-stone-600">Model details load from the prediction service.</p>
      ) : (
        <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
          <div>
            <dt className="text-stone-600">Model name</dt>
            <dd>{modelInfo.model_name}</dd>
          </div>
          <div>
            <dt className="text-stone-600">Model family</dt>
            <dd>{modelInfo.model_family}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-stone-600">Model version</dt>
            <dd className="break-all">{modelInfo.model_version}</dd>
          </div>
          <div>
            <dt className="text-stone-600">Prediction target</dt>
            <dd>{modelInfo.prediction_target}</dd>
          </div>
          <div>
            <dt className="text-stone-600">Calibration</dt>
            <dd>{modelInfo.calibration}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-stone-600">Prediction horizon</dt>
            <dd>{modelInfo.prediction_horizon.description}</dd>
          </div>
          <div>
            <dt className="text-stone-600">Supported inputs</dt>
            <dd>{modelInfo.supported_parameters.length}</dd>
          </div>
          <div>
            <dt className="text-stone-600">Research early-warning threshold</dt>
            <dd>{formatThreshold(modelInfo.decision_threshold)}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt className="text-stone-600">Training dataset</dt>
            <dd>{modelInfo.training_dataset_version}</dd>
          </div>
        </dl>
      )}
    </section>
  );
}
