// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { StatusPage } from "./StatusPage";
import "./styles.css";

const container = document.getElementById("root");
if (container === null) throw new Error("#root element missing from status/index.html");

createRoot(container).render(
  <StrictMode>
    <StatusPage />
  </StrictMode>,
);
