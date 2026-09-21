import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";
import { App } from "./App";
import "@fontsource/barlow/latin-400.css";
import "@fontsource/barlow/latin-500.css";
import "@fontsource/barlow/latin-600.css";
import "@fontsource/barlow-semi-condensed/latin-400.css";
import "@fontsource/barlow-semi-condensed/latin-500.css";
import "@fontsource/barlow-semi-condensed/latin-600.css";
import "./styles.css";

// Browsers fetch a face only when some text first needs it, so text arriving later (a clip card, a heading) would
// flash in the fallback face. Ask for every face the interface uses up front; they are local and small.
for (const family of ["Barlow", '"Barlow Semi Condensed"']) {
  for (const weight of [400, 500, 600]) void document.fonts?.load(`${weight} 1em ${family}`);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
);
