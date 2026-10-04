"use client";

import { useEffect, useState } from "react";
import { AboutModel } from "@/components/AboutModel";
import { DataSufficiencyCard } from "@/components/DataSufficiencyCard";
import { Disclaimer } from "@/components/Disclaimer";
import { ErrorState } from "@/components/ErrorState";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { HistoryEditor } from "@/components/HistoryEditor";
import { HistoryTable } from "@/components/HistoryTable";
import { LoadingState } from "@/components/LoadingState";
import { PredictionForm } from "@/components/PredictionForm";
import { RiskCard } from "@/components/RiskCard";
import { RiskTrendChart } from "@/components/RiskTrendChart";
import { ServiceStatus } from "@/components/ServiceStatus";
import { ApiError, getHealth, getModelInfo, predict, predictSequence } from "@/lib/api";
import { currentRequest, sequenceRequest } from "@/lib/observations";
import { emptyFields } from "@/lib/parameters";
import type { CurrentPrediction, FieldValues, ModelInfo, SequencePrediction } from "@/lib/types";

type Mode = "current" | "history";

function latestHourPrediction(result: SequencePrediction): CurrentPrediction {
  const hour = result.predictions[result.predictions.length - 1];
  return {
    ...hour,
    prediction_horizon: result.prediction_horizon,
    model_name: result.model_name,
    model_version: result.model_version,
    disclaimer: result.disclaimer,
  };
}

export function Dashboard() {
  const [mode, setMode] = useState<Mode>("current");
  const [checking, setChecking] = useState(true);
  const [online, setOnline] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [currentValues, setCurrentValues] = useState<FieldValues>(emptyFields);
  const [hours, setHours] = useState<FieldValues[]>([emptyFields()]);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<CurrentPrediction | null>(null);
  const [historyResult, setHistoryResult] = useState<SequencePrediction | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const health = await getHealth();
      if (cancelled) return;
      setOnline(health.online);
      setStatusMessage(health.message);
      setChecking(false);
      if (!health.online) {
        setModelInfo(null);
        return;
      }
      try {
        const info = await getModelInfo();
        if (!cancelled) {
          setModelInfo(info);
          setModelError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setModelError(error instanceof ApiError ? error.message : "Model information is unavailable.");
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const latestHistory = historyResult?.predictions.at(-1) ?? null;
  const sufficiencySource = mode === "history" ? latestHistory : currentResult;
  const explanations = mode === "history" ? latestHistory?.explanations ?? null : currentResult?.explanations ?? null;
  const disclaimer = mode === "history" ? historyResult?.disclaimer : currentResult?.disclaimer;
  const serviceBlocked = !online || submitting;

  async function estimateCurrent() {
    if (!online || submitting) return;
    const request = currentRequest(currentValues);
    setFieldErrors(request.errors);
    if (Object.keys(request.errors).length > 0) {
      setRequestError("Correct the highlighted fields. Empty fields can stay empty.");
      return;
    }
    setSubmitting(true);
    setRequestError(null);
    try {
      setCurrentResult(await predict(request.body));
    } catch (error) {
      setRequestError(error instanceof ApiError ? error.message : "The prediction service could not complete the request.");
    } finally {
      setSubmitting(false);
    }
  }

  async function analyzeHistory() {
    if (!online || submitting) return;
    const request = sequenceRequest(hours);
    setFieldErrors(request.errors);
    if (Object.keys(request.errors).length > 0) {
      setRequestError("Correct the highlighted fields. Empty fields can stay empty.");
      return;
    }
    setSubmitting(true);
    setRequestError(null);
    try {
      setHistoryResult(await predictSequence(request.body));
    } catch (error) {
      setRequestError(error instanceof ApiError ? error.message : "The prediction service could not complete the request.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen">
      <a href="#clinical-input" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-10 focus:bg-white focus:px-3 focus:py-2">
        Skip to clinical input
      </a>
      <header className="border-b border-line bg-panel">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-tideink">Research prototype</p>
            <h1 className="text-2xl font-semibold">Sepsis Early-Warning System</h1>
            <p className="text-sm text-stone-700">Dynamic early sepsis risk prediction</p>
          </div>
          <ServiceStatus online={online} checking={checking} />
        </div>
      </header>
      <main className="mx-auto grid min-w-0 max-w-6xl gap-6 px-4 py-6">
        <Disclaimer apiDisclaimer={disclaimer ?? modelInfo?.disclaimer} />
        {!online && !checking ? <ErrorState message={statusMessage ?? "Prediction service: Unavailable"} /> : null}
        <div className="grid min-w-0 items-start gap-6 lg:grid-cols-2">
          <section id="clinical-input" className="min-w-0 space-y-4" aria-label="Clinical input">
            <div className="flex flex-wrap gap-2" role="tablist" aria-label="Input mode">
              <button
                type="button"
                role="tab"
                aria-selected={mode === "current"}
                className={`rounded-md px-3 py-2 text-sm font-semibold ${mode === "current" ? "bg-tide text-white" : "border border-line bg-white"}`}
                onClick={() => setMode("current")}
              >
                Current estimate
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === "history"}
                className={`rounded-md px-3 py-2 text-sm font-semibold ${mode === "history" ? "bg-tide text-white" : "border border-line bg-white"}`}
                onClick={() => setMode("history")}
              >
                Hourly history
              </button>
            </div>
            {mode === "current" ? (
              <PredictionForm
                values={currentValues}
                errors={fieldErrors}
                disabled={serviceBlocked}
                onChange={(name, value) => setCurrentValues((current) => ({ ...current, [name]: value }))}
                onSubmit={() => void estimateCurrent()}
              />
            ) : (
              <HistoryEditor
                hours={hours}
                errors={fieldErrors}
                disabled={serviceBlocked}
                onChange={(index, name, value) =>
                  setHours((current) => current.map((hour, hourIndex) => (hourIndex === index ? { ...hour, [name]: value } : hour)))
                }
                onAdd={() => setHours((current) => [...current, emptyFields()])}
                onRemove={(index) => setHours((current) => current.filter((_, hourIndex) => hourIndex !== index))}
                onSubmit={() => void analyzeHistory()}
              />
            )}
          </section>
          <section className="min-w-0 space-y-4 lg:sticky lg:top-4" aria-label="Prediction result">
            {submitting ? <LoadingState /> : null}
            {requestError ? <ErrorState message={requestError} /> : null}
            <RiskCard
              prediction={mode === "current" ? currentResult : historyResult ? latestHourPrediction(historyResult) : null}
              modelInfo={modelInfo}
              caption={mode === "history" && historyResult ? "Latest submitted hour" : undefined}
            />
          </section>
        </div>
        <RiskTrendChart rows={historyResult?.predictions ?? null} />
        <HistoryTable rows={historyResult?.predictions ?? null} />
        <div className="grid min-w-0 gap-6 lg:grid-cols-2">
          <DataSufficiencyCard
            sufficiency={sufficiencySource?.data_sufficiency ?? null}
            available={sufficiencySource?.data_available ?? null}
            total={sufficiencySource?.data_total ?? null}
            dynamicCount={sufficiencySource?.observed_dynamic_variables ?? null}
            laboratoryCount={sufficiencySource?.observed_laboratory_variables ?? null}
            historicalHours={sufficiencySource?.historical_hours ?? null}
            hasPrior={sufficiencySource?.has_prior_measurements ?? null}
          />
          <ExplanationPanel explanations={explanations} />
        </div>
        <AboutModel modelInfo={modelInfo} error={modelError} />
      </main>
    </div>
  );
}
