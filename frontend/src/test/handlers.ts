import { http, HttpResponse } from "msw";
import { CATEGORY, inspection } from "./fixtures";

export const BASE = "http://localhost:8000/api/v1";

/** The default answers: a ready service with one category. Tests override them per case. */
export const handlers = [
  http.get(`${BASE}/categories`, () => HttpResponse.json([CATEGORY])),
  http.post(`${BASE}/inspections`, () => HttpResponse.json(inspection())),
];
