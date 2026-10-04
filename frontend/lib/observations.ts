import type { FieldValues, ObservationRequest } from "@/lib/types";

const DECIMAL = /^[+-]?(?:\d+\.?\d*|\.\d+)$/;

export type ParsedField =
  | { state: "empty" }
  | { state: "invalid" }
  | { state: "number"; value: number };

export function parseField(raw: string | undefined): ParsedField {
  const trimmed = (raw ?? "").trim();
  if (trimmed === "") return { state: "empty" };
  if (!DECIMAL.test(trimmed)) return { state: "invalid" };
  const value = Number(trimmed);
  if (!Number.isFinite(value)) return { state: "invalid" };
  return { state: "number", value };
}

export function collectValues(fields: FieldValues): {
  values: Record<string, number>;
  errors: Record<string, string>;
} {
  const values: Record<string, number> = {};
  const errors: Record<string, string> = {};
  for (const [name, raw] of Object.entries(fields)) {
    const parsed = parseField(raw);
    if (parsed.state === "empty") continue;
    if (parsed.state === "invalid") {
      errors[name] = "Enter a number, or leave this field empty.";
      continue;
    }
    values[name] = parsed.value;
  }
  return { values, errors };
}

export function currentRequest(fields: FieldValues): {
  errors: Record<string, string>;
  body: ObservationRequest;
} {
  const { values, errors } = collectValues(fields);
  return { errors, body: { observations: [{ values }] } };
}

export function sequenceRequest(hours: FieldValues[]): {
  errors: Record<string, string>;
  body: ObservationRequest;
} {
  const errors: Record<string, string> = {};
  const observations = hours.map((fields, index) => {
    const collected = collectValues(fields);
    for (const [name, message] of Object.entries(collected.errors)) {
      errors[`${index}:${name}`] = message;
    }
    return { values: collected.values };
  });
  return { errors, body: { observations } };
}
