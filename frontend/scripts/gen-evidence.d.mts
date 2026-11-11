export interface Evidence {
  source: string;
  source_sha256: string;
  category: string;
  version: string;
  backbone: string;
  n_anomalous: number;
  n_normal: number;
  defects_flagged: number;
  defects_missed: number;
  false_alarms: number;
  false_alarm_rate: number;
  target_rate: number;
}
export const METRICS_RELATIVE: string;
export const METRICS_PATH: string;
export const SNAPSHOT_PATH: string;
export function buildEvidence(metricsText: string): Evidence;
export function snapshotText(metricsText: string): string;
