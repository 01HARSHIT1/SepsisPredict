import { PARAMETER_LABELS } from "@/lib/parameters";
import type { Explanation } from "@/lib/types";

export function ExplanationPanel({ explanations }: { explanations: Explanation[] | null }) {
  return (
    <section className="rounded-lg border border-line bg-panel p-5" aria-label="Factors influencing this estimate">
      <h2 className="text-lg font-semibold">Factors influencing this estimate</h2>
      <p className="mt-2 text-sm text-stone-700">This variable contributed to the model&apos;s current estimate.</p>
      {!explanations ? (
        <p className="mt-3 text-sm text-stone-600">Shown after a prediction is returned.</p>
      ) : explanations.length === 0 ? (
        <p className="mt-3 text-sm">The model returned no variable-level contributions for this estimate.</p>
      ) : (
        <ol className="mt-3 space-y-2">
          {explanations.slice(0, 6).map((item) => (
            <li key={item.feature} className="flex items-start justify-between gap-3 rounded-md border border-line px-3 py-2 text-sm">
              <span className="font-medium">{PARAMETER_LABELS[item.feature] ?? item.feature}</span>
              <span>
                {item.direction === "increases_risk_score" ? "Higher model contribution" : "Lower model contribution"}
              </span>
            </li>
          ))}
        </ol>
      )}
      <p className="mt-3 text-xs text-stone-600">
        Model explanations describe statistical contribution to the prediction and are not clinical causal explanations.
      </p>
    </section>
  );
}
