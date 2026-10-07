import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App.tsx"
import "./globals.css"
import { ThemeProvider } from "next-themes"
import { TooltipProvider } from "./components/ui/tooltip.tsx"

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider defaultTheme="dark" disableTransitionOnChange>
      <TooltipProvider>
        <App />
      </TooltipProvider>
    </ThemeProvider>
  </React.StrictMode>
)
