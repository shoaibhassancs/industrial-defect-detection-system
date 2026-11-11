import { useEffect } from "react";

/** Sets the browser tab title for the current page (screen readers and history use it). */
export function useDocumentTitle(title: string): void {
  useEffect(() => {
    document.title = title === "DefectLens" ? title : `${title} · DefectLens`;
  }, [title]);
}
