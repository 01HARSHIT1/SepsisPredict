import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HistoryTable } from "@/components/HistoryTable";
import { RiskCard } from "@/components/RiskCard";
import { predict, predictSequence } from "@/lib/api";
import { formatRiskPercent } from "@/lib/format";

const live = process.env.RUN_LIVE_API === "1";

describe.skipIf(!live)("live FastAPI model", () => {
  it("shows the current probability returned by the saved model", async () => {
    const result = await predict({
      observations: [{ values: { HR: 118, Temp: 38.7, MAP: 64, Resp: 25 } }],
    });
    render(<RiskCard prediction={result} modelInfo={null} />);
    expect(screen.getByTestId("risk-percent")).toHaveTextContent(formatRiskPercent(result.risk_probability));
    expect(result.model_name).toBe("flexible_xgboost");
    expect(result.data_sufficiency).toBe("MODERATE");
    expect(result.research_early_warning).toEqual(expect.any(Boolean));
  });

  it("shows one returned probability per historical hour", async () => {
    const result = await predictSequence({
      observations: [
        { values: { HR: 88, MAP: 82, Temp: 37.1 } },
        { values: { HR: 94, MAP: 78, Temp: 37.4 } },
        { values: { HR: 105, MAP: 71, Temp: 38.0 } },
        { values: { HR: 116, MAP: 64, Temp: 38.6 } },
      ],
    });
    render(<HistoryTable rows={result.predictions} />);
    expect(result.predictions).toHaveLength(4);
    expect(result.model_name).toBe("flexible_xgboost");
    for (const row of result.predictions) {
      expect(screen.getAllByText(formatRiskPercent(row.risk_probability)).length).toBeGreaterThan(0);
    }
    expect(result.predictions[0].risk_trend).toBeNull();
    expect(result.predictions[3].historical_hours).toBe(4);
  });
});
