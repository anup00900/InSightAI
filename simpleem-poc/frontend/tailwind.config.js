/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0e1a',
        'bg-secondary': '#111827',
        'bg-card': '#1a2332',
        'bg-card-hover': '#1e2a3d',
        'border': '#2a3654',
        'border-light': '#374766',
        'accent': '#3b82f6',
        'accent-light': '#60a5fa',
        'accent-glow': '#3b82f640',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'text-primary': '#f1f5f9',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
