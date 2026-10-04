import { formatSufficiency } from "@/lib/format";
import type { DataSufficiency } from "@/lib/types";

export function DataSufficiencyCard({
  sufficiency,
  available,
  total,
  dynamicCount,
  laboratoryCount,
  historicalHours,
  hasPrior,
}: {
  sufficiency: DataSufficiency | null;
  available: number | null;
  total: number | null;
  dynamicCount: number | null;
  laboratoryCount: number | null;
  historicalHours: number | null;
  hasPrior: boolean | null;
}) {
  return (
    <section className="rounded-lg border border-line bg-panel p-5" aria-label="Data sufficiency">
      <h2 className="text-lg font-semibold">Data Sufficiency</h2>
      <p className="mt-2 text-sm text-stone-700">
        Indicates how much information was available to the model. It is not clinical confidence.
      </p>
      {sufficiency ? (
        <p className="mt-3 text-2xl font-semibold" data-testid="sufficiency-band">
          {formatSufficiency(sufficiency)}
        </p>
      ) : (
        <p className="mt-3 text-sm text-stone-600">Shown after a prediction is returned.</p>
      )}
      {available !== null && total !== null ? (
        <ul className="mt-3 space-y-1 text-sm">
          <li>
            Parameters present this hour: {available} / {total}
          </li>
          <li>Observed vital or laboratory values: {dynamicCount}</li>
          <li>Observed laboratory values: {laboratoryCount}</li>
          <li>Historical hours: {historicalHours}</li>
          <li>Earlier measurements in this submission: {hasPrior ? "yes" : "no"}</li>
        </ul>
      ) : null}
    </section>
  );
}
