import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/errors";

export type Loadable<T> =
  | { status: "loading"; waking: boolean }
  | { status: "ready"; data: T }
  | { status: "error"; error: ApiError };

const MAX_AUTO_RETRIES = 24;

/** Runs `load` once on mount. While the model is still loading (503 MODEL_NOT_READY) it asks again
 * by itself, because that state ends without anyone doing anything; every other error stops and
 * leaves a reload function for a button. `load` must be a stable function (declared once). */
export function useStartupRetry<T>(
  load: (signal: AbortSignal) => Promise<T>,
  defaultRetryMs = 5000,
): { state: Loadable<T>; reload: () => void } {
  const [state, setState] = useState<Loadable<T>>({ status: "loading", waking: false });
  const [attempt, setAttempt] = useState(0);
  const retries = useRef(0);

  const reload = useCallback(() => {
    retries.current = 0;
    setState({ status: "loading", waking: false });
    setAttempt((n) => n + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    load(controller.signal)
      .then((data) => setState({ status: "ready", data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const apiError =
          error instanceof ApiError
            ? error
            : new ApiError({ code: "BAD_RESPONSE", message: String(error) });
        if (apiError.code === "MODEL_NOT_READY" && retries.current < MAX_AUTO_RETRIES) {
          retries.current += 1;
          setState({ status: "loading", waking: true });
          const wait =
            apiError.retryAfterSeconds !== null
              ? apiError.retryAfterSeconds * 1000
              : defaultRetryMs;
          timer = setTimeout(() => setAttempt((n) => n + 1), wait);
        } else {
          setState({ status: "error", error: apiError });
        }
      });
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [load, attempt, defaultRetryMs]);

  return { state, reload };
}
