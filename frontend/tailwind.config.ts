import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        tui: {
          bg: "#0a0a0a",
          fg: "#e5e5e5",
          dim: "#737373",
          accent: "#10b981",
          warn: "#f59e0b",
          err: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
export default config;
