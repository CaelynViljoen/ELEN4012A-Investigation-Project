// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3490dc',
        secondary: '#ffed4a',
        danger: '#e3342f',
        background: '#f7fafc',
      },
      borderRadius: {
        'xl': '1rem',
        'br-lg': '1.5rem 1.5rem 0 0',
      },
      backdropBlur: {
        xs: '2px',
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
        '2xl': '24px',
        '3xl': '40px',
      },
      backgroundOpacity: {
        10: '0.1',
        20: '0.2',
        30: '0.3',
        40: '0.4',
        50: '0.5',
        60: '0.6',
        70: '0.7',
        80: '0.8',
        90: '0.9',
      },
      fontFamily: {
        martel: ['Martel', 'serif'],
        assistant: ['Assistant', 'sans-serif'],
      },
    },
  },
  variants: {
    extend: {
      backdropBlur: ['responsive'],
      backgroundOpacity: ['responsive'],
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.module.rules.push({
        test: /\.mjs$/,
        enforce: 'pre',
        use: ['source-map-loader'],
        exclude: [/node_modules\/@mediapipe\/tasks-vision/], // Exclude problematic package
      });
      return webpackConfig;
    },
  },
};
