import React from "react";
import ReactDOM from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { LorsaPage } from "@/routes/lorsa/page";
import { RootPage } from "./routes/page";

const router = createBrowserRouter([
  {
    path: "/lorsa",
    element: <LorsaPage />,
  },
  {
    path: "/",
    element: <RootPage />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
