import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aiden — Your job search, together",
  description: "A personal assistant for your next career move.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="de">
      <body>{children}</body>
    </html>
  );
}
