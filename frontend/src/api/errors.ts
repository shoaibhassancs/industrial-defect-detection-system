/** Codes the backend sends (see backend/app/core/errors.py) plus three the browser adds. */
export type ErrorCode =
  | "VALIDATION_ERROR"
  | "UNSUPPORTED_CATEGORY"
  | "UNSUPPORTED_MEDIA_TYPE"
  | "FILE_TOO_LARGE"
  | "INVALID_IMAGE"
  | "IMAGE_DIMENSIONS"
  | "MODEL_NOT_READY"
  | "BUSY"
  | "RATE_LIMITED"
  | "INFERENCE_FAILED"
  | "INTERNAL_ERROR"
  | "NOT_FOUND"
  | "METHOD_NOT_ALLOWED"
  | "HTTP_ERROR"
  | "NETWORK" // the request never got an answer (server down, or the browser blocked it)
  | "BAD_RESPONSE" // an answer arrived but it was not in the documented shape
  | "ABORTED"; // the user cancelled

/** One failed call, in one shape, whatever went wrong. */
export class ApiError extends Error {
  readonly code: ErrorCode | string;
  readonly status: number | null;
  readonly details: Record<string, unknown>;
  readonly requestId: string | null;
  /** Seconds, from the Retry-After header, when the server gave one. */
  readonly retryAfterSeconds: number | null;

  constructor(init: {
    code: ErrorCode | string;
    message: string;
    status?: number | null;
    details?: Record<string, unknown>;
    requestId?: string | null;
    retryAfterSeconds?: number | null;
  }) {
    super(init.message);
    this.name = "ApiError";
    this.code = init.code;
    this.status = init.status ?? null;
    this.details = init.details ?? {};
    this.requestId = init.requestId ?? null;
    this.retryAfterSeconds = init.retryAfterSeconds ?? null;
  }
}

/** Plain-language sentence for each code. The server's own message is kept as a fallback. */
export function describeError(
  error: ApiError,
  apiBaseUrl: string,
): { title: string; hint: string } {
  switch (error.code) {
    case "UNSUPPORTED_MEDIA_TYPE":
      return {
        title: "That file is not a JPEG or PNG image.",
        hint: "Choose a .jpg or .png file.",
      };
    case "FILE_TOO_LARGE": {
      const mb = error.details["max_mb"];
      return {
        title: `That image is too large${typeof mb === "number" ? ` (limit ${mb} MB)` : ""}.`,
        hint: "Choose a smaller image or save it at a lower quality.",
      };
    }
    case "INVALID_IMAGE":
      return {
        title: "The image could not be read.",
        hint: "The file may be damaged or cut off. Try saving it again.",
      };
    case "IMAGE_DIMENSIONS":
      return {
        title: "The image size is outside what this demo accepts.",
        hint: "Very small and very large images are refused. Try one of ordinary photo size.",
      };
    case "UNSUPPORTED_CATEGORY":
      return {
        title: "That category is not available.",
        hint: "Only the categories listed in the menu can be inspected.",
      };
    case "VALIDATION_ERROR":
      return {
        title: "The request was incomplete.",
        hint: "Choose a category and an image, then try again.",
      };
    case "RATE_LIMITED":
      return {
        title: "Too many requests in a short time.",
        hint:
          error.retryAfterSeconds !== null
            ? `Please wait about ${error.retryAfterSeconds} seconds and try again.`
            : "Please wait a minute and try again.",
      };
    case "BUSY":
      return {
        title: "The service is busy with another image.",
        hint: "Please try again in a few seconds.",
      };
    case "MODEL_NOT_READY":
      return {
        title: "The model is still starting up.",
        hint: "This can take a little while on a free host. Please try again shortly.",
      };
    case "INFERENCE_FAILED":
      return {
        title: "The model could not process this image.",
        hint: "Try a different image. If it keeps happening, report the request ID below.",
      };
    case "INTERNAL_ERROR":
      return {
        title: "Something went wrong on the server.",
        hint: "Please try again. If it keeps happening, report the request ID below.",
      };
    case "NETWORK":
      return {
        title: "The service could not be reached.",
        hint: `Checked ${apiBaseUrl}. The backend may not be running, or it may not allow this site's address (CORS). Your browser does not tell the page which of the two it is.`,
      };
    case "BAD_RESPONSE":
      return {
        title: "The service answered in an unexpected way.",
        hint: "This is a bug or a version mismatch between the site and the service.",
      };
    case "ABORTED":
      return { title: "Cancelled.", hint: "" };
    default:
      return { title: error.message || "Something went wrong.", hint: "Please try again." };
  }
}
