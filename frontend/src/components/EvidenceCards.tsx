import { percent } from "../format";

export interface EvidenceCardsProps {
  /** Defective test images the demo flagged, and how many defective test images there were. */
  defectsFlagged: number;
  defectiveTotal: number;
  /** Normal test images wrongly flagged, how many normal test images there were, and the saved rate. */
  falseAlarms: number;
  normalTotal: number;
  falseAlarmRate: number;
  /** The rate the threshold aimed for (alpha). */
  targetRate: number;
  /** The threshold, shown on the Results page only. */
  threshold?: string;
}

/** The two headline results side by side. Used by the Home page (saved snapshot) and the Results
 * page (API values), so both always say the same thing. Nothing is calculated except one
 * comparison of two stored rates, which picks "higher than" or "not higher than". */
export function EvidenceCards(p: EvidenceCardsProps) {
  const higher = p.falseAlarmRate > p.targetRate;
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <div className="rounded-lg border-2 border-emerald-700 bg-emerald-50 p-4">
        <p className="text-sm font-medium">Defective test images flagged</p>
        <p className="text-2xl font-semibold">
          {p.defectsFlagged} of {p.defectiveTotal}
        </p>
        {p.threshold && <p className="text-sm">at the calibrated threshold {p.threshold}.</p>}
      </div>
      <div className="rounded-lg border-2 border-amber-700 bg-amber-50 p-4">
        <p className="text-sm font-medium">Normal test images wrongly flagged (false alarms)</p>
        <p className="text-2xl font-semibold">
          {p.falseAlarms} of {p.normalTotal} ({percent(p.falseAlarmRate)})
        </p>
        <p className="text-sm">
          {higher
            ? `That is higher than the ${percent(p.targetRate)} the threshold aimed for.`
            : `That is not higher than the ${percent(p.targetRate)} the threshold aimed for.`}
        </p>
      </div>
    </div>
  );
}
