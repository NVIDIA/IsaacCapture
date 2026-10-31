import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const container = document.getElementById("root");
if (container === null) throw new Error("#root element missing from index.html");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
