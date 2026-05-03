/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      colors: {
        // CodeLens dark theme
        'dark': {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
          950: '#0f1419',
        },
        // Primary gradient
        'primary': {
          50: '#f0f4ff',
          500: '#667eea',
          600: '#5568d3',
          700: '#4657ba',
          800: '#3c4a9e',
        }
      },
      fontFamily: {
        mono: ['Fira Code', 'Courier New', 'monospace'],
      },
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },
      animation: {
        'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;',
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
