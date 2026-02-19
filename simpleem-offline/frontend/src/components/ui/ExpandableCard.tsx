import { useState, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { clsx } from 'clsx';

interface ExpandableCardProps {
  children: ReactNode;
  expandedContent: ReactNode;
  className?: string;
  glowColor?: string;
}

export default function ExpandableCard({
  children,
  expandedContent,
  className,
  glowColor = 'rgba(139,92,246,0.15)',
}: ExpandableCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className={clsx(
        'glass-card cursor-pointer transition-all duration-300 select-none',
        expanded
          ? 'bg-white/[0.08] border-white/20'
          : 'hover:bg-white/[0.07] hover:border-white/15 hover:-translate-y-0.5',
        className,
      )}
      style={{
        boxShadow: expanded ? `0 0 25px ${glowColor}, 0 4px 20px rgba(0,0,0,0.3)` : 'none',
      }}
    >
      <div className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex-1">{children}</div>
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
            className="ml-2 mt-1"
          >
            <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
          </motion.div>
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-0">
              <div className="border-t border-white/10 pt-4">
                {expandedContent}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
