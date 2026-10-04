export function Disclaimer({ apiDisclaimer }: { apiDisclaimer?: string | null }) {
  return (
    <aside className="rounded-lg border border-line bg-panel p-4 text-sm leading-6" aria-label="Research disclaimer">
      <p>
        This application is a research/educational early-warning prototype based on the PhysioNet 2019
        SepsisLabel. It does not diagnose sepsis and must not be used as a substitute for clinical assessment.
      </p>
      {apiDisclaimer ? <p className="mt-2 text-stone-700">{apiDisclaimer}</p> : null}
    </aside>
  );
}
