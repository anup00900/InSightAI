import { ReactNode, ButtonHTMLAttributes } from 'react';
import { clsx } from 'clsx';
import { Loader2 } from 'lucide-react';

interface GradientButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: ReactNode;
}

export default function GradientButton({
  children, variant = 'primary', size = 'md', loading, icon, className, disabled, ...props
}: GradientButtonProps) {
  const variants = {
    primary: 'bg-gradient-primary hover:shadow-glow-purple',
    secondary: 'bg-gradient-secondary hover:shadow-glow-blue',
    success: 'bg-gradient-success hover:shadow-glow-success',
    ghost: 'bg-transparent border border-border hover:border-accent hover:bg-bg-card',
  };
  const sizes = {
    sm: 'px-3 py-1.5 text-xs rounded-lg',
    md: 'px-5 py-2.5 text-sm rounded-xl',
    lg: 'px-8 py-3.5 text-base rounded-xl',
  };

  return (
    <button
      className={clsx(
        'font-semibold text-white transition-all duration-300 flex items-center gap-2',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}
