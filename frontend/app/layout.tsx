import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sepsis Early-Warning System",
  description: "Research prototype for dynamic early sepsis risk estimates. Not a diagnosis.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-paper font-sans text-ink antialiased">{children}</body>
    </html>
  );
}
