import { ParameterGroup } from "@/components/ParameterGroup";
import { PARAMETER_GROUPS } from "@/lib/parameters";

export function PredictionForm({
  values,
  errors,
  disabled,
  onChange,
  onSubmit,
}: {
  values: Record<string, string>;
  errors: Record<string, string>;
  disabled: boolean;
  onChange: (name: string, value: string) => void;
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
        Enter any subset of the supported parameters. An empty field is omitted. A typed 0 is sent as a measured zero.
      </p>
      {PARAMETER_GROUPS.map((group) => (
        <ParameterGroup
          key={group.id}
          group={group}
          values={values}
          errors={errors}
          idPrefix="current"
          onChange={onChange}
        />
      ))}
      <button
        type="submit"
        className="rounded-md bg-tide px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-stone-400"
        disabled={disabled}
      >
        Estimate Sepsis Risk
      </button>
    </form>
  );
}
