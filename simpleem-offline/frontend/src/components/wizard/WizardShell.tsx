import { ReactNode } from 'react';
import { Check, Upload, FileText, BarChart3, LayoutDashboard } from 'lucide-react';
import { clsx } from 'clsx';

interface WizardShellProps {
  currentStep: number; // 1-4
  children: ReactNode;
}

const STEPS = [
  { num: 1, label: 'Upload', icon: Upload },
  { num: 2, label: 'Transcribe', icon: FileText },
  { num: 3, label: 'Analyze', icon: BarChart3 },
  { num: 4, label: 'Results', icon: LayoutDashboard },
];

export default function WizardShell({ currentStep, children }: WizardShellProps) {
  return (
    <div className="space-y-8">
      {/* Step indicator */}
      <div className="flex items-center justify-center gap-0">
        {STEPS.map((step, idx) => {
          const isActive = step.num === currentStep;
          const isComplete = step.num < currentStep;
          const Icon = step.icon;
          return (
            <div key={step.num} className="flex items-center">
              <div className="flex flex-col items-center">
                <div className={clsx(
                  'w-12 h-12 rounded-full flex items-center justify-center transition-all duration-500 border-2',
                  isComplete && 'bg-gradient-to-br from-emerald-500 to-cyan-500 border-emerald-500/50 shadow-glow-success',
                  isActive && 'bg-gradient-to-br from-indigo-500 to-purple-500 border-purple-500/50 shadow-glow-purple animate-glow-pulse',
                  !isActive && !isComplete && 'bg-white/5 border-white/10 text-text-muted',
                )}>
                  {isComplete ? (
                    <Check className="w-5 h-5 text-white" />
                  ) : (
                    <Icon className={clsx('w-5 h-5', isActive ? 'text-white' : 'text-text-muted')} />
                  )}
                </div>
                <span className={clsx(
                  'mt-2 text-xs font-medium',
                  isActive ? 'text-accent-light' : isComplete ? 'text-success' : 'text-text-muted',
                )}>
                  {step.label}
                </span>
              </div>
              {idx < STEPS.length - 1 && (
                <div className={clsx(
                  'w-20 h-0.5 mx-3 mt-[-20px] rounded-full transition-all duration-500',
                  step.num < currentStep ? 'bg-gradient-to-r from-emerald-500 to-cyan-500' : 'bg-white/10',
                )} />
              )}
            </div>
          );
        })}
      </div>

      {/* Content */}
      <div className="animate-fade-in">
        {children}
      </div>
    </div>
  );
}
