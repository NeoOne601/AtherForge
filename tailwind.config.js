/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./frontend/src/**/*.{ts,tsx}", "./frontend/index.html"],
    darkMode: "class",
    theme: {
        extend: {
            fontFamily: {
                sans: ["Inter", "system-ui", "sans-serif"],
                mono: ["JetBrains Mono", "Fira Code", "monospace"],
            },
            colors: {
                // AetherForge brand palette — deep space + electric accent
                aether: {
                    950: "#020408",
                    900: "#050c14",
                    800: "#0a1628",
                    700: "#0f2040",
                    600: "#162d58",
                    500: "#1e3d78",
                    400: "#2856a8",
                    300: "#3b72d4",
                    200: "#6497e0",
                    100: "#a0bfef",
                    50: "#dde9f9",
                },
                volt: {
                    600: "#7c3aed",
                    500: "#8b5cf6",
                    400: "#a78bfa",
                    300: "#c4b5fd",
                },
                plasma: {
                    500: "#06b6d4",
                    400: "#22d3ee",
                    300: "#67e8f9",
                },
                ember: {
                    500: "#f97316",
                    400: "#fb923c",
                },
                safe: {
                    500: "#10b981",
                    400: "#34d399",
                },
                danger: {
                    500: "#ef4444",
                    400: "#f87171",
                },
            },
            animation: {
                "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
                "glow": "glow 2s ease-in-out infinite alternate",
                "slide-up": "slideUp 0.3s ease-out",
                "fade-in": "fadeIn 0.2s ease-out",
            },
            keyframes: {
                glow: {
                    "0%": { boxShadow: "0 0 5px #8b5cf6, 0 0 10px #8b5cf6" },
                    "100%": { boxShadow: "0 0 10px #8b5cf6, 0 0 25px #8b5cf6, 0 0 50px #6d28d9" },
                },
                slideUp: {
                    "0%": { opacity: "0", transform: "translateY(10px)" },
                    "100%": { opacity: "1", transform: "translateY(0)" },
                },
                fadeIn: {
                    "0%": { opacity: "0" },
                    "100%": { opacity: "1" },
                },
            },
            backdropBlur: {
                xs: "2px",
            },
        },
    },
    plugins: [],
};
