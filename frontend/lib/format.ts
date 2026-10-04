import type { DataSufficiency, RiskTrend } from "@/lib/types";

export function formatRiskPercent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

export function formatCompleteness(completeness: number): string {
  return `${(completeness * 100).toFixed(0)}%`;
}

export function formatSufficiency(value: DataSufficiency): string {
  return value.replaceAll("_", " ");
}

export function formatTrend(trend: RiskTrend): string | null {
  if (trend === "increasing") return "Risk increasing";
  if (trend === "decreasing") return "Risk decreasing";
  if (trend === "unchanged") return "Stable";
  return null;
}

export function formatThreshold(threshold: number): string {
  return `${(threshold * 100).toFixed(1)}%`;
}
