import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DataSufficiencyCard } from "@/components/DataSufficiencyCard";
import { ErrorState } from "@/components/ErrorState";
import { RiskCard } from "@/components/RiskCard";
import { ServiceStatus } from "@/components/ServiceStatus";
import { Dashboard } from "@/components/Dashboard";
import type { CurrentPrediction } from "@/lib/types";

const prediction: CurrentPrediction = {
  risk_probability: 0.137,
  research_early_warning: true,
  data_available: 4,
  data_total: 40,
  data_completeness: 0.1,
  data_sufficiency: "MODERATE",
  data_sufficiency_rule: "dynamic_variables_at_this_hour_v1",
  historical_hours: 1,
  observed_dynamic_variables: 4,
  observed_laboratory_variables: 0,
  has_prior_measurements: false,
  risk_trend: null,
  prediction_horizon: { name: "official_sepsis_label", description: "Hourly probability of the PhysioNet 2019 SepsisLabel." },
  model_version: "phase5-test",
  model_name: "flexible_xgboost",
  explanations: [],
  disclaimer: "Research estimate. Not a diagnosis.",
};

describe("prediction display", () => {
  it("renders the probability returned by the caller", () => {
    render(<RiskCard prediction={prediction} modelInfo={null} />);
    expect(screen.getByTestId("risk-percent")).toHaveTextContent("13.7%");
    expect(screen.getByText(/not a diagnosis/i)).toBeInTheDocument();
    expect(screen.queryByText(/definitely has sepsis/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/no sepsis/i)).not.toBeInTheDocument();
  });

  it("does not show a fallback probability before a response exists", () => {
    render(<RiskCard prediction={null} modelInfo={null} />);
    expect(screen.queryByTestId("risk-percent")).not.toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it("renders data sufficiency as an information category", () => {
    render(
      <DataSufficiencyCard
        sufficiency="LOW"
        available={1}
        total={40}
        dynamicCount={1}
        laboratoryCount={0}
        historicalHours={1}
        hasPrior={false}
      />,
    );
    expect(screen.getByTestId("sufficiency-band")).toHaveTextContent("LOW");
    expect(screen.getByText(/not clinical confidence/i)).toBeInTheDocument();
    expect(screen.queryByText(/model confidence/i)).not.toBeInTheDocument();
  });

  it("renders a backend error message", () => {
    render(<ErrorState message="The prediction service did not accept the request." />);
    expect(screen.getByRole("alert")).toHaveTextContent("did not accept");
  });

  it("renders an unavailable prediction service", () => {
    render(<ServiceStatus online={false} checking={false} />);
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
  });
});

vi.mock("@/lib/api", () => ({
  getHealth: vi.fn(async () => ({
    online: false,
    message: "Unable to reach the prediction service. Please make sure the backend is running.",
  })),
  getModelInfo: vi.fn(),
  predict: vi.fn(),
  predictSequence: vi.fn(),
  ApiError: class ApiError extends Error {
    status: number | null = null;
  },
}));

describe("dashboard without a backend", () => {
  it("shows the unavailable state and does not invent a probability", async () => {
    const api = await import("@/lib/api");
    render(<Dashboard />);
    expect(await screen.findByText(/unable to reach the prediction service/i)).toBeInTheDocument();
    expect(screen.queryByTestId("risk-percent")).not.toBeInTheDocument();
    expect(api.predict).not.toHaveBeenCalled();
  });
});
