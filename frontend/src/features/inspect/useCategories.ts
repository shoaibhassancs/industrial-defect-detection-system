import { getCategories } from "../../api/client";
import type { ApiError } from "../../api/errors";
import type { CategoryInfo } from "../../api/types";
import { useStartupRetry } from "./useStartupRetry";

export type CategoriesState =
  | { status: "loading"; waking: boolean }
  | { status: "ready"; categories: CategoryInfo[] }
  | { status: "error"; error: ApiError };

/** The category list, with the shared "wait for a starting model" behaviour. */
export function useCategories(defaultRetryMs = 5000): {
  state: CategoriesState;
  reload: () => void;
} {
  const { state, reload } = useStartupRetry(getCategories, defaultRetryMs);
  if (state.status === "ready") {
    return { state: { status: "ready", categories: state.data }, reload };
  }
  return { state, reload };
}
