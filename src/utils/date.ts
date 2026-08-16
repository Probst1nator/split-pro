/**
 * Parse a "YYYY-MM-DD" date as local midnight. `new Date('2026-03-14')` is an ISO instant
 * (2026-03-14T00:00:00Z), so browsers west of UTC render it as the previous day. Anything
 * else is handed to `Date` unchanged.
 */
export const parseDateOnly = (value: string): Date => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value.trim());
  if (!match) {
    return new Date(value);
  }
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
};
