/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#08090e',
        'bg-secondary': '#0d0e14',
        'bg-card': 'rgba(255,255,255,0.04)',
        'bg-card-hover': 'rgba(255,255,255,0.06)',
        'bg-card-solid': '#101118',
        'border': 'rgba(255,255,255,0.08)',
        'border-light': 'rgba(255,255,255,0.12)',
        'border-glow': 'rgba(56,189,248,0.3)',
        'accent': '#38bdf8',
        'accent-light': '#7dd3fc',
        'accent-blue': '#38bdf8',
        'accent-cyan': '#22d3ee',
        'accent-glow': 'rgba(56,189,248,0.3)',
        'success': '#10b981',
        'success-glow': 'rgba(16,185,129,0.3)',
        'warning': '#f59e0b',
        'warning-glow': 'rgba(245,158,11,0.3)',
        'danger': '#ef4444',
        'danger-glow': 'rgba(239,68,68,0.3)',
        'text-primary': '#e2e8f0',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0,0,0,0.5)',
        'glow-purple': '0 0 15px rgba(139,92,246,0.2)',
        'glow-blue': '0 0 20px rgba(56,189,248,0.2)',
        'glow-cyan': '0 0 15px rgba(34,211,238,0.2)',
        'glow-success': '0 0 15px rgba(16,185,129,0.3)',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #38bdf8, #818cf8, #c084fc)',
        'gradient-secondary': 'linear-gradient(135deg, #22d3ee, #38bdf8)',
        'gradient-success': 'linear-gradient(135deg, #10b981, #06b6d4)',
        'gradient-warm': 'linear-gradient(135deg, #f59e0b, #ef4444)',
      },
    },
  },
  plugins: [],
}
