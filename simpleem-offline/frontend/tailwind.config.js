/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0a1a',
        'bg-secondary': '#0f0f2e',
        'bg-card': 'rgba(255,255,255,0.05)',
        'bg-card-hover': 'rgba(255,255,255,0.08)',
        'bg-card-solid': '#141430',
        'border': 'rgba(255,255,255,0.1)',
        'border-light': 'rgba(255,255,255,0.15)',
        'border-glow': 'rgba(139,92,246,0.3)',
        'accent': '#8b5cf6',
        'accent-light': '#a78bfa',
        'accent-blue': '#3b82f6',
        'accent-cyan': '#06b6d4',
        'accent-glow': 'rgba(139,92,246,0.4)',
        'success': '#10b981',
        'success-glow': 'rgba(16,185,129,0.3)',
        'warning': '#f59e0b',
        'warning-glow': 'rgba(245,158,11,0.3)',
        'danger': '#ef4444',
        'danger-glow': 'rgba(239,68,68,0.3)',
        'text-primary': '#f1f5f9',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0,0,0,0.3)',
        'glow-purple': '0 0 15px rgba(139,92,246,0.3)',
        'glow-blue': '0 0 15px rgba(59,130,246,0.3)',
        'glow-cyan': '0 0 15px rgba(6,182,212,0.3)',
        'glow-success': '0 0 15px rgba(16,185,129,0.3)',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7)',
        'gradient-secondary': 'linear-gradient(135deg, #06b6d4, #3b82f6)',
        'gradient-success': 'linear-gradient(135deg, #10b981, #06b6d4)',
        'gradient-warm': 'linear-gradient(135deg, #f59e0b, #ef4444)',
      },
    },
  },
  plugins: [],
}
