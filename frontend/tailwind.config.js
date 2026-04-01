/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', 'sans-serif'],
      },
      colors: {
        bg: '#070d1f',
        surface: {
          DEFAULT: '#0c1326',
          raised: '#11192e',
          high: '#171f36',
          highest: '#1c253e',
        },
        primary: {
          DEFAULT: '#818cf8',
          dim: '#6366f1',
          glow: 'rgba(129,140,248,0.15)',
        },
        secondary: '#a855f7',
        risk: {
          critical: '#ef4444',
          dangerous: '#f97316',
          high: '#eab308',
          moderate: '#f59e0b',
          low: '#22c55e',
        },
        outline: 'rgba(255,255,255,0.08)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'primary-gradient': 'linear-gradient(135deg, #818cf8, #a855f7)',
        'danger-gradient': 'linear-gradient(135deg, #ef4444, #f97316)',
      },
      animation: {
        'shimmer': 'shimmer 2s infinite linear',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'fade-up': 'fadeUp 0.5s ease-out forwards',
        'scale-in': 'scaleIn 0.3s ease-out forwards',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(129,140,248,0.1)' },
          '50%': { boxShadow: '0 0 40px rgba(129,140,248,0.3)' },
        },
        fadeUp: {
          from: { opacity: 0, transform: 'translateY(16px)' },
          to: { opacity: 1, transform: 'translateY(0)' },
        },
        scaleIn: {
          from: { opacity: 0, transform: 'scale(0.95)' },
          to: { opacity: 1, transform: 'scale(1)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
