export const formatMoney = (value, currency = 'USD') => {
  if (value == null) return '';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(value);
};

export const formatSeconds = (sec) => {
  if (sec == null) return '';
  const d = Number(sec);
  const h = Math.floor(d / 3600);
  const m = Math.floor((d % 3600) / 60);
  const s = Math.floor(d % 3600 % 60);
  const hDisplay = h > 0 ? h + (h === 1 ? 'h ' : 'h ') : '';
  const mDisplay = m > 0 ? m + (m === 1 ? 'm ' : 'm ') : '';
  const sDisplay = s > 0 ? s + (s === 1 ? 's' : 's') : '';
  return (hDisplay + mDisplay + sDisplay).trim() || '0s';
};
