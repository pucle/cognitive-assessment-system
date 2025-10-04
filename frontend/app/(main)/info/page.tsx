import { Metadata } from "next";
import InfoPageClient from "./InfoPageClient";

export const metadata: Metadata = {
  title: "Cá Vàng: Thắp Sáng Ký Ức | AI Phân Tích Giọng Nói Sàng Lọc Sa Sút Trí Tuệ",
  description: "Hệ thống AI tiên tiến phân tích giọng nói để sàng lọc sa sút trí tuệ tại nhà. Phát triển bởi đội học sinh THPT Chuyên Lê Quý Đôn với độ chính xác 94.2%.",
  keywords: [
    "sa sút trí tuệ",
    "dementia",
    "AI",
    "phân tích giọng nói",
    "sàng lọc tại nhà",
    "Cá Vàng",
    "THPT Chuyên Lê Quý Đôn",
    "MCI",
    "Alzheimer's",
    "công nghệ y tế"
  ],
  authors: [
    { name: "Đội Cá Vàng", url: "https://cavang.info" },
    { name: "THPT Chuyên Lê Quý Đôn" }
  ],
  creator: "Đội Cá Vàng",
  publisher: "THPT Chuyên Lê Quý Đôn",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://cavang.info'),
  alternates: {
    canonical: '/info',
  },
  openGraph: {
    title: "Cá Vàng: Thắp Sáng Ký Ức | AI Sàng Lọc Sa Sút Trí Tuệ",
    description: "Hệ thống AI phân tích giọng nói để phát hiện sớm sa sút trí tuệ. Độ chính xác 94.2%, dễ sử dụng tại nhà.",
    url: '/info',
    siteName: 'Cá Vàng: Thắp Sáng Ký Ức',
    images: [
      {
        url: '/og-image-info.png',
        width: 1200,
        height: 630,
        alt: 'Cá Vàng - AI Phân tích giọng nói sàng lọc sa sút trí tuệ'
      }
    ],
    locale: 'vi_VN',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: "Cá Vàng: Thắp Sáng Ký Ức | AI Sàng Lọc Sa Sút Trí Tuệ",
    description: "Hệ thống AI phân tích giọng nói để phát hiện sớm sa sút trí tuệ. Độ chính xác 94.2%, dễ sử dụng tại nhà.",
    images: ['/og-image-info.png'],
  },
  robots: {
    index: true,
    follow: true,
    nocache: true,
    googleBot: {
      index: true,
      follow: true,
      noimageindex: false,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-site-verification-code',
  },
};

export default function InfoPage() {
  return <InfoPageClient />;
}
