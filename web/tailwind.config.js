/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // The chrome is hue-less on purpose: the frame and its class-coloured mask are the only colour on screen,
      // so nothing around them skews how they read. `key` (keyframes) and `danger` are the two exceptions.
      colors: {
        surround: "#2B2D30",
        well: "#1E1F21",
        panel: "#36383C",
        raise: "#44474C",
        lift: "#52555B",
        line: "#4A4D52",
        ink: "#E9E7E2",
        dim: "#A3A5AA",
        key: "#F59E0B",
        danger: "#FF8A80",
      },
      fontFamily: {
        sans: ["Barlow", "system-ui", "sans-serif"],
        cond: ['"Barlow Semi Condensed"', "Barlow", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
