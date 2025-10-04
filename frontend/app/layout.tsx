import "./globals.css";
import { type Metadata } from 'next'
import { ClientProviders } from './providers/ClientProviders';
import { Nunito } from "next/font/google";

const font = Nunito({
  variable: "--font-nunito",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Cá Vàng",
  description: "Ứng dụng chăm sóc trí nhớ",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi">
      <body className={`${font.variable} font-sans antialiased`}>
        <ClientProviders>
          {children}
        </ClientProviders>
      </body>
    </html>
  );
}
