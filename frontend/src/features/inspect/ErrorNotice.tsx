import { apiBaseUrl } from "../../api/client";
import { type ApiError, describeError } from "../../api/errors";

export function ErrorNotice({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const { title, hint } = describeError(error, apiBaseUrl());
  return (
    <div role="alert" className="rounded-lg border-2 border-red-700 bg-red-50 p-4 text-red-950">
      <p className="font-semibold">
        <span aria-hidden="true">⚠ </span>
        {title}
      </p>
      {hint && <p className="mt-1 text-sm">{hint}</p>}
      {error.requestId && (
        <p className="mt-2 text-xs">
          Request ID: <code className="font-mono">{error.requestId}</code>
        </p>
      )}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded border border-red-800 bg-white px-3 py-1.5 text-sm font-medium text-red-900 hover:bg-red-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-800"
        >
          Try again
        </button>
      )}
    </div>
  );
}
