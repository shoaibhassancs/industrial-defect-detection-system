// Display formatting only. These functions change how a stored number is written (decimal places,
// percent sign); they never compute a new metric.

/** A stored rate or score written with a fixed number of decimals, e.g. 0.9545 -> "0.955". */
export function decimals(value: number, places = 3): string {
  return value.toFixed(places);
}

/** A stored fraction written as a percentage without trailing zeros: 0.15 -> "15%", 0.0465 -> "4.65%". */
export function percent(fraction: number, places = 2): string {
  return `${Number((fraction * 100).toFixed(places))}%`;
}

/** "broken_large" -> "broken large". */
export function humanize(name: string): string {
  return name.replace(/_/g, " ");
}
