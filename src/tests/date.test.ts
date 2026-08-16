import { parseDateOnly } from '~/utils/date';

describe('parseDateOnly', () => {
  it('parses a date-only string as local midnight, not UTC midnight', () => {
    const date = parseDateOnly('2026-03-14');

    expect(date.getFullYear()).toBe(2026);
    expect(date.getMonth()).toBe(2);
    expect(date.getDate()).toBe(14);
    expect(date.getHours()).toBe(0);
  });

  it('keeps the calendar day regardless of the runtime offset', () => {
    // `new Date('2026-01-01')` is 2026-01-01T00:00:00Z, which is 2025-12-31 anywhere west of UTC.
    expect(parseDateOnly('2026-01-01').getDate()).toBe(1);
    expect(parseDateOnly('2026-01-01').getMonth()).toBe(0);
  });

  it('tolerates surrounding whitespace', () => {
    expect(parseDateOnly(' 2026-03-14 ').getDate()).toBe(14);
  });

  it('falls back to Date for anything that is not date-only', () => {
    const withTime = parseDateOnly('2026-03-14T12:30:00Z');

    expect(withTime.toISOString()).toBe('2026-03-14T12:30:00.000Z');
    expect(isNaN(parseDateOnly('not a date').getTime())).toBe(true);
  });
});
