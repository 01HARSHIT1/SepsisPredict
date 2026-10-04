import type { CurrentPrediction, ModelInfo, ObservationRequest, SequencePrediction } from "@/lib/types";

const TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function apiBaseUrl(): string {
  const value = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (!value) {
    throw new ApiError("The prediction service address is not configured.", null);
  }
  return value.replace(/\/$/, "");
}

export function detailMessage(detail: unknown): string | null {
  if (typeof detail === "string") {
    if (detail.length > 0 && detail.length < 300 && !detail.includes("Traceback")) return detail;
    return null;
  }
  if (!Array.isArray(detail)) return null;
  const messages = detail
    .map((item) => {
      if (!item || typeof item !== "object" || !("msg" in item)) return "";
      return String((item as { msg: unknown }).msg);
    })
    .filter((message) => message.length > 0 && !message.includes("Traceback"));
  const text = messages.slice(0, 3).join(" ");
  if (text.length > 0 && text.length < 300) return text;
  return null;
}

async function readFailure(response: Response): Promise<string> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  const detail =
    payload && typeof payload === "object" && "detail" in payload
      ? (payload as { detail: unknown }).detail
      : null;
  if (response.status === 400 || response.status === 422) {
    return detailMessage(detail) ?? "The prediction service did not accept the request.";
  }
  if (response.status === 503) return "The prediction service is unavailable.";
  if (response.status >= 500) return "The prediction service had a problem completing the request.";
  return "The prediction service could not complete the request.";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl()}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("Unable to reach the prediction service. Please make sure the backend is running.", null);
  }
  if (!response.ok) {
    throw new ApiError(await readFailure(response), response.status);
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("The prediction service returned a response that could not be read.", response.status);
  }
}

function requireProbability(value: unknown, status: number | null): asserts value is { risk_probability: number } {
  if (!value || typeof value !== "object" || typeof (value as { risk_probability?: unknown }).risk_probability !== "number") {
    throw new ApiError("The prediction service returned a response that could not be read.", status);
  }
}

export async function getHealth(): Promise<{ online: boolean; message: string | null }> {
  try {
    const body = await request<{ status?: string }>("/health");
    if (body.status === "ok") return { online: true, message: null };
    return { online: false, message: "The prediction service did not report a healthy status." };
  } catch (error) {
    const message = error instanceof ApiError ? error.message : "Unable to reach the prediction service. Please make sure the backend is running.";
    return { online: false, message };
  }
}

export function getModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/api/model-info");
}

export async function predict(body: ObservationRequest): Promise<CurrentPrediction> {
  const result = await request<CurrentPrediction>("/api/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
  requireProbability(result, 200);
  return result;
}

export async function predictSequence(body: ObservationRequest): Promise<SequencePrediction> {
  const result = await request<SequencePrediction>("/api/predict-sequence", {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (!Array.isArray(result.predictions) || result.predictions.some((row) => typeof row.risk_probability !== "number")) {
    throw new ApiError("The prediction service returned a response that could not be read.", 200);
  }
  return result;
}
