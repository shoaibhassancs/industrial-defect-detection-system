/** A native expandable section. Closed by default; the text stays in the page and can be opened
 * with the keyboard. Used for supporting detail, never for the key evidence or the limitations
 * that must stay visible. */
export function Detail({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <details className={`rounded border border-slate-300 p-3 ${className}`}>
      <summary className="cursor-pointer font-medium">{title}</summary>
      <div className="mt-3 space-y-3 text-sm text-slate-800">{children}</div>
    </details>
  );
}
