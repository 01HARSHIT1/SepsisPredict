import { formatRiskPercent, formatTrend } from "@/lib/format";
import type { SequenceHourPrediction } from "@/lib/types";

export function RiskTrendChart({ rows }: { rows: SequenceHourPrediction[] | null }) {
  return (
    <section className="min-w-0 rounded-lg border border-line bg-panel p-5" aria-label="Risk trend">
      <h2 className="text-lg font-semibold">Risk trend</h2>
      {!rows || rows.length === 0 ? (
        <p className="mt-2 text-sm text-stone-600">Available after an hourly history analysis.</p>
      ) : (
        <Chart rows={rows} />
      )}
    </section>
  );
}

function Chart({ rows }: { rows: SequenceHourPrediction[] }) {
  const width = 640;
  const height = 220;
  const pad = 28;
  const values = rows.map((row) => row.risk_probability * 100);
  const max = Math.max(1, ...values);
  const points = values.map((value, index) => {
    const x = pad + (rows.length === 1 ? (width - pad * 2) / 2 : (index / (rows.length - 1)) * (width - pad * 2));
    const y = height - pad - (value / max) * (height - pad * 2);
    return { x, y, value, row: rows[index] };
  });
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const summary = points
    .map((point, index) => `Hour ${index + 1} ${formatRiskPercent(point.row.risk_probability)}`)
    .join(", ");

  return (
    <div className="mt-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-56 w-full" role="img" aria-label={`Estimated sepsis risk by hour: ${summary}`}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#a8a29e" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#a8a29e" />
        <path d={path} fill="none" stroke="#0f766e" strokeWidth="2" />
        {points.map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.y} r="4" fill="#134e4a" />
            <text x={point.x} y={height - 8} textAnchor="middle" fontSize="12" fill="#44403c">
              {index + 1}
            </text>
            <text x={point.x} y={point.y - 8} textAnchor="middle" fontSize="12" fill="#1c1917">
              {formatRiskPercent(point.row.risk_probability)}
            </text>
          </g>
        ))}
      </svg>
      <p className="text-xs text-stone-600">Hour on the horizontal axis. Estimated sepsis risk (%) on the vertical axis.</p>
      <ul className="mt-2 space-y-1 text-sm">
        {rows.map((row, index) => {
          const trend = formatTrend(row.risk_trend);
          return (
            <li key={index}>
              Hour {index + 1}: {formatRiskPercent(row.risk_probability)}
              {trend ? ` · ${trend}` : ""}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
