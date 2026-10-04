import { ParameterGroup } from "@/components/ParameterGroup";
import { PARAMETER_GROUPS } from "@/lib/parameters";
import type { FieldValues } from "@/lib/types";

export function HistoryEditor({
  hours,
  errors,
  disabled,
  onChange,
  onAdd,
  onRemove,
  onSubmit,
}: {
  hours: FieldValues[];
  errors: Record<string, string>;
  disabled: boolean;
  onChange: (index: number, name: string, value: string) => void;
  onAdd: () => void;
  onRemove: (index: number) => void;
  onSubmit: () => void;
}) {
  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <p className="text-sm text-stone-700">
        Hours are sent in the order shown, from Hour 1 downward. Each hour can contain a different subset of parameters.
      </p>
      {hours.map((values, index) => (
        <section key={index} className="space-y-3 rounded-lg border border-line bg-stone-50 p-3" aria-label={`Hour ${index + 1}`}>
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-base font-semibold">Hour {index + 1}</h3>
            <button
              type="button"
              className="rounded-md border border-line bg-white px-3 py-1 text-sm disabled:cursor-not-allowed disabled:text-stone-400"
              onClick={() => onRemove(index)}
              disabled={disabled || hours.length === 1}
            >
              Remove Hour
            </button>
          </div>
          {PARAMETER_GROUPS.map((group) => (
            <details key={group.id} open={group.id === "vitals"} className="rounded-md bg-white">
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium">{group.title}</summary>
              <div className="px-1 pb-2">
                <ParameterGroup
                  group={group}
                  values={values}
                  errors={errorsForHour(errors, index)}
                  idPrefix={`hour-${index}`}
                  showLegend={false}
                  onChange={(name, value) => onChange(index, name, value)}
                />
              </div>
            </details>
          ))}
        </section>
      ))}
      <div className="flex flex-wrap gap-3">
        <button type="button" className="rounded-md border border-line bg-white px-4 py-2 text-sm font-semibold" onClick={onAdd} disabled={disabled}>
          Add Hour
        </button>
        <button
          type="submit"
          className="rounded-md bg-tide px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-stone-400"
          disabled={disabled}
        >
          Analyze History
        </button>
      </div>
    </form>
  );
}

function errorsForHour(errors: Record<string, string>, index: number): Record<string, string> {
  const prefix = `${index}:`;
  const local: Record<string, string> = {};
  for (const [key, message] of Object.entries(errors)) {
    if (key.startsWith(prefix)) local[key.slice(prefix.length)] = message;
  }
  return local;
}
