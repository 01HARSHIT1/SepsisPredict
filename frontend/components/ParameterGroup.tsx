import type { ParameterGroup as ParameterGroupModel } from "@/lib/parameters";

export function ParameterGroup({
  group,
  values,
  errors,
  idPrefix,
  onChange,
  showLegend = true,
}: {
  group: ParameterGroupModel;
  values: Record<string, string>;
  errors: Record<string, string>;
  idPrefix: string;
  onChange: (name: string, value: string) => void;
  showLegend?: boolean;
}) {
  return (
    <fieldset className="rounded-lg border border-line bg-white p-4">
      {showLegend ? <legend className="px-1 text-sm font-semibold">{group.title}</legend> : <legend className="sr-only">{group.title}</legend>}
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {group.fields.map((field) => {
          const id = `${idPrefix}-${field.name}`;
          const error = errors[field.name];
          const describedBy = error ? `${id}-error` : `${id}-unit`;
          return (
            <div key={field.name}>
              <label htmlFor={id} className="block text-sm font-medium">
                {field.label}
              </label>
              {field.kind === "binary" ? (
                <select
                  id={id}
                  className="mt-1 w-full rounded-md border border-line bg-panel px-2 py-2 text-sm"
                  value={values[field.name] ?? ""}
                  aria-invalid={Boolean(error)}
                  aria-describedby={describedBy}
                  onChange={(event) => onChange(field.name, event.target.value)}
                >
                  <option value="">Not entered</option>
                  {field.options?.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  id={id}
                  className="mt-1 w-full rounded-md border border-line bg-panel px-2 py-2 text-sm"
                  inputMode="decimal"
                  autoComplete="off"
                  value={values[field.name] ?? ""}
                  aria-invalid={Boolean(error)}
                  aria-describedby={describedBy}
                  onChange={(event) => onChange(field.name, event.target.value)}
                />
              )}
              <p id={`${id}-unit`} className="mt-1 text-xs text-stone-600">
                {field.unit}. Leave empty to omit.
              </p>
              {error ? (
                <p id={`${id}-error`} className="mt-1 text-xs text-amber-900">
                  {error}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
    </fieldset>
  );
}
