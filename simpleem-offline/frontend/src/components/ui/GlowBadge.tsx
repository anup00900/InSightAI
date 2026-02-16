import { clsx } from 'clsx';

interface GlowBadgeProps {
  children: React.ReactNode;
  variant?: 'purple' | 'blue' | 'cyan' | 'success' | 'warning' | 'danger' | 'neutral';
  size?: 'sm' | 'md';
}

const VARIANTS = {
  purple: 'bg-accent/20 text-accent-light border-accent/30',
  blue: 'bg-accent-blue/20 text-blue-400 border-accent-blue/30',
  cyan: 'bg-accent-cyan/20 text-cyan-400 border-accent-cyan/30',
  success: 'bg-success/20 text-emerald-400 border-success/30',
  warning: 'bg-warning/20 text-amber-400 border-warning/30',
  danger: 'bg-danger/20 text-red-400 border-danger/30',
  neutral: 'bg-white/5 text-text-secondary border-white/10',
};

export default function GlowBadge({ children, variant = 'neutral', size = 'sm' }: GlowBadgeProps) {
  return (
    <span className={clsx(
      'inline-flex items-center font-medium border rounded-full',
      VARIANTS[variant],
      size === 'sm' ? 'px-2 py-0.5 text-[10px]' : 'px-3 py-1 text-xs',
    )}>
      {children}
    </span>
  );
}
