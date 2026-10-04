import { formatCompleteness, formatRiskPercent, formatSufficiency, formatTrend } from "@/lib/format";
import type { SequenceHourPrediction } from "@/lib/types";

export function HistoryTable({ rows }: { rows: SequenceHourPrediction[] | null }) {
  return (
    <section className="min-w-0 rounded-lg border border-line bg-panel p-5" aria-label="Hourly results">
      <h2 className="text-lg font-semibold">Hourly results</h2>
      {!rows || rows.length === 0 ? (
        <p className="mt-2 text-sm text-stone-600">Shown after an hourly history analysis.</p>
      ) : (
        <div className="mt-3 min-w-0 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <caption className="sr-only">Estimated risk for each submitted hour</caption>
            <thead>
              <tr className="border-b border-line">
                <th className="py-2 pr-4 font-semibold">Hour</th>
                <th className="py-2 pr-4 font-semibold">Risk</th>
                <th className="py-2 pr-4 font-semibold">Available parameters</th>
                <th className="py-2 pr-4 font-semibold">Completeness</th>
                <th className="py-2 pr-4 font-semibold">Data sufficiency</th>
                <th className="py-2 font-semibold">Risk trend</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index} className="border-b border-line">
                  <td className="py-2 pr-4">Hour {index + 1}</td>
                  <td className="py-2 pr-4 tabular-nums">{formatRiskPercent(row.risk_probability)}</td>
                  <td className="py-2 pr-4">
                    {row.data_available} / {row.data_total}
                  </td>
                  <td className="py-2 pr-4">{formatCompleteness(row.data_completeness)}</td>
                  <td className="py-2 pr-4">{formatSufficiency(row.data_sufficiency)}</td>
                  <td className="py-2">{formatTrend(row.risk_trend) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
