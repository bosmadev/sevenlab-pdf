import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "Cascadia Code",
          "SF Mono",
          "ui-monospace",
          "monospace",
        ],
      },
      colors: {
        tui: {
          base: "#0b0e11",
          surface: "#12161c",
          elevated: "#1a1f28",
          overlay: "#232a35",
          border: "#2a3040",
          borderHover: "#3d4a5c",
          borderActive: "#94a3b8",
          text: "#c9d1d9",
          textMuted: "#929faf",
          textBright: "#e6edf3",
          accent: "#94a3b8",
          success: "#4ade80",
          danger: "#f87171",
          warn: "#fbbf24",
          info: "#60a5fa",
        },
      },
      borderRadius: {
        tui: "6px",
      },
    },
  },
  plugins: [],
};
export default config;
