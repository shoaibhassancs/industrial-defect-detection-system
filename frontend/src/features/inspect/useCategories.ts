import { useCallback, useEffect, useRef, useState } from "react";
import { getCategories } from "../../api/client";
import { ApiError } from "../../api/errors";
import type { CategoryInfo } from "../../api/types";

export type CategoriesState =
  | { status: "loading"; waking: boolean }
  | { status: "ready"; categories: CategoryInfo[] }
  | { status: "error"; error: ApiError };

const MAX_AUTO_RETRIES = 24;

/** Loads the category list. While the model is still loading (503 MODEL_NOT_READY) it asks again by
 * itself, because that state ends without anyone doing anything; every other error stops and shows
 * a button instead. */
export function useCategories(defaultRetryMs = 5000): {
  state: CategoriesState;
  reload: () => void;
} {
  const [state, setState] = useState<CategoriesState>({ status: "loading", waking: false });
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
    getCategories(controller.signal)
      .then((categories) => setState({ status: "ready", categories }))
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
  }, [attempt, defaultRetryMs]);

  return { state, reload };
}
