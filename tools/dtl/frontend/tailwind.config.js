/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      colors: {
        eng: {
          bg: "var(--bg-app)",
          panel: "var(--bg-panel)",
          "panel-secondary": "var(--bg-panel-secondary)",
          "panel-inset": "var(--bg-panel-inset)",
          border: "var(--border-subtle)",
          "border-muted": "var(--border-muted)",
          accent: "var(--accent)",
          "accent-hover": "var(--accent-hover)",
          "accent-subtle": "var(--accent-subtle)",
        },
      },
    },
  },
  plugins: [],
};
