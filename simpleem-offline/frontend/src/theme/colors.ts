export const THEME = {
  bg: {
    primary: '#0a0a1a',
    secondary: '#0f0f2e',
    card: 'rgba(255,255,255,0.05)',
  },
  accent: {
    purple: '#8b5cf6',
    blue: '#3b82f6',
    cyan: '#06b6d4',
    indigo: '#6366f1',
  },
  gradient: {
    primary: ['#6366f1', '#8b5cf6', '#a855f7'],
    secondary: ['#06b6d4', '#3b82f6'],
    success: ['#10b981', '#06b6d4'],
  },
  status: {
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
  },
  text: {
    primary: '#f1f5f9',
    secondary: '#94a3b8',
    muted: '#64748b',
  },
  chart: {
    colors: ['#8b5cf6', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899'],
    grid: 'rgba(255,255,255,0.06)',
    tooltip: {
      bg: 'rgba(15,15,46,0.95)',
      border: 'rgba(255,255,255,0.1)',
    },
  },
} as const;

export const SPEAKER_COLORS = [
  '#8b5cf6', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#ef4444', '#14b8a6',
];

export function getScoreColor(score: number): string {
  if (score >= 80) return '#10b981';
  if (score >= 60) return '#3b82f6';
  if (score >= 40) return '#f59e0b';
  return '#ef4444';
}

export function getScoreGlow(score: number): string {
  if (score >= 80) return 'shadow-glow-success';
  if (score >= 60) return 'shadow-glow-blue';
  if (score >= 40) return 'shadow-glow-purple';
  return '';
}
