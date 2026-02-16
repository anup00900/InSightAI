import { ReactNode } from 'react';
import { clsx } from 'clsx';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
  padding?: boolean;
}

export default function GlassCard({ children, className, hover = false, glow = false, padding = true }: GlassCardProps) {
  return (
    <div className={clsx(
      'glass-card',
      hover && 'glass-card-hover transition-all duration-300 cursor-pointer',
      glow && 'animate-glow-pulse',
      padding && 'p-5',
      className,
    )}>
      {children}
    </div>
  );
}
