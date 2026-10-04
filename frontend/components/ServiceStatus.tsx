export function ServiceStatus({ online, checking }: { online: boolean; checking: boolean }) {
  const label = checking ? "Checking" : online ? "Online" : "Unavailable";
  return (
    <p className="flex items-center gap-2 text-sm" aria-live="polite">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${checking ? "bg-stone-400" : online ? "bg-tide" : "bg-stone-700"}`}
        aria-hidden="true"
      />
      <span>
        Prediction service: <strong>{label}</strong>
      </span>
    </p>
  );
}
