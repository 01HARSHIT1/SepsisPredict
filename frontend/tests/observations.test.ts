import { describe, expect, it } from "vitest";
import { currentRequest, sequenceRequest } from "@/lib/observations";
import { emptyFields } from "@/lib/parameters";

describe("observation requests", () => {
  it("omits empty optional fields and preserves a typed zero", () => {
    const fields = emptyFields();
    fields.HR = "118";
    fields.Temp = "38.7";
    fields.MAP = "64";
    fields.Resp = "25";
    fields.Glucose = "0";
    fields.Lactate = "   ";
    fields.WBC = "";

    const { errors, body } = currentRequest(fields);

    expect(errors).toEqual({});
    expect(body).toEqual({
      observations: [
        {
          values: {
            HR: 118,
            Temp: 38.7,
            MAP: 64,
            Resp: 25,
            Glucose: 0,
          },
        },
      ],
    });
    expect(body.observations[0].values).not.toHaveProperty("Lactate");
    expect(body.observations[0].values).not.toHaveProperty("WBC");
    expect(JSON.stringify(body)).not.toContain("null");
  });

  it("rejects non-numeric text without turning it into zero", () => {
    const fields = emptyFields();
    fields.HR = "fast";
    const { errors, body } = currentRequest(fields);
    expect(errors.HR).toMatch(/number/i);
    expect(body.observations[0].values).not.toHaveProperty("HR");
    expect(body.observations[0].values.HR).not.toBe(0);
  });

  it("preserves hourly order", () => {
    const first = emptyFields();
    const second = emptyFields();
    const third = emptyFields();
    first.HR = "88";
    first.MAP = "82";
    second.HR = "94";
    third.HR = "116";
    third.Temp = "38.6";

    const { body } = sequenceRequest([first, second, third]);

    expect(body.observations.map((item) => item.values.HR)).toEqual([88, 94, 116]);
    expect(body.observations[0].values.MAP).toBe(82);
    expect(body.observations[1].values).not.toHaveProperty("MAP");
    expect(body.observations[2].values.Temp).toBe(38.6);
    expect(body.observations.map((item) => "hour" in item)).toEqual([false, false, false]);
  });
});
