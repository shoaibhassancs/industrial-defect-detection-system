import { ApiError } from "./errors";
import { type MetricsInfo, type ModelInfo, parseMetrics, parseModelInfo } from "./modelTypes";
import type { CategoryInfo, InspectionResponse } from "./types";

const DEFAULT_BASE_URL = "http://localhost:8000";

/** Where the API lives. Set VITE_API_BASE_URL at build time; no trailing slash needed. */
export function apiBaseUrl(): string {
  const configured = import.meta.env["VITE_API_BASE_URL"] as string | undefined;
  return (configured && configured.trim() !== "" ? configured : DEFAULT_BASE_URL).replace(
    /\/+$/,
    "",
  );
}

const PREFIX = "/api/v1";

function retryAfter(response: Response): number | null {
  const raw = response.headers.get("Retry-After");
  const seconds = raw === null ? NaN : Number(raw);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Turn any non-2xx answer into an ApiError. Works for the documented envelope and for anything else
 * (for example an HTML error page from a proxy), and never throws. */
async function errorFrom(response: Response): Promise<ApiError> {
  const requestId = response.headers.get("X-Request-ID");
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  const inner = isRecord(body) && isRecord(body["error"]) ? body["error"] : null;
  if (inner && typeof inner["code"] === "string" && typeof inner["message"] === "string") {
    return new ApiError({
      code: inner["code"],
      message: inner["message"],
      status: response.status,
      details: isRecord(inner["details"]) ? inner["details"] : {},
      requestId: typeof inner["request_id"] === "string" ? inner["request_id"] : requestId,
      retryAfterSeconds: retryAfter(response),
    });
  }
  return new ApiError({
    code: "HTTP_ERROR",
    message: `The service answered with HTTP ${response.status}.`,
    status: response.status,
    requestId,
    retryAfterSeconds: retryAfter(response),
  });
}

async function send(url: string, init: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    if (init.signal?.aborted) {
      throw new ApiError({ code: "ABORTED", message: "Cancelled." });
    }
    throw new ApiError({
      code: "NETWORK",
      message: cause instanceof Error ? cause.message : "Network error",
    });
  }
  if (!response.ok) throw await errorFrom(response);
  return response;
}

async function jsonOf(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new ApiError({
      code: "BAD_RESPONSE",
      message: "The answer was not valid JSON.",
      status: response.status,
    });
  }
}

function isInspection(value: unknown): value is InspectionResponse {
  if (!isRecord(value)) return false;
  const images = value["images"];
  return (
    typeof value["prediction"] === "string" &&
    typeof value["anomaly_score"] === "number" &&
    typeof value["threshold"] === "number" &&
    isRecord(value["calibration_rank"]) &&
    isRecord(images) &&
    isRecord(images["original"]) &&
    isRecord(images["heatmap"]) &&
    isRecord(images["overlay"])
  );
}

function isCategoryList(value: unknown): value is CategoryInfo[] {
  return (
    Array.isArray(value) &&
    value.every(
      (c) => isRecord(c) && typeof c["id"] === "string" && typeof c["display_name"] === "string",
    )
  );
}

/** GET /api/v1/categories */
export async function getCategories(signal?: AbortSignal): Promise<CategoryInfo[]> {
  const response = await send(`${apiBaseUrl()}${PREFIX}/categories`, { signal });
  const body = await jsonOf(response);
  if (!isCategoryList(body)) {
    throw new ApiError({ code: "BAD_RESPONSE", message: "Unexpected categories answer." });
  }
  return body;
}

/** POST /api/v1/inspections (multipart). The file name is not sent as anything meaningful. */
export async function inspectImage(
  category: string,
  file: File,
  signal?: AbortSignal,
): Promise<InspectionResponse> {
  const form = new FormData();
  form.append("category", category);
  form.append("file", file, file.name);
  const response = await send(`${apiBaseUrl()}${PREFIX}/inspections`, {
    method: "POST",
    body: form,
    signal,
  });
  const body = await jsonOf(response);
  if (!isInspection(body)) {
    throw new ApiError({
      code: "BAD_RESPONSE",
      message: "Unexpected inspection answer.",
      status: response.status,
    });
  }
  return body;
}

/** GET /api/v1/model */
export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const response = await send(`${apiBaseUrl()}${PREFIX}/model`, { signal });
  return parseModelInfo(await jsonOf(response));
}

/** GET /api/v1/metrics/{category}: the saved sealed-evaluation results, as the package holds them. */
export async function getMetrics(category: string, signal?: AbortSignal): Promise<MetricsInfo> {
  const response = await send(`${apiBaseUrl()}${PREFIX}/metrics/${encodeURIComponent(category)}`, {
    signal,
  });
  return parseMetrics(await jsonOf(response));
}
