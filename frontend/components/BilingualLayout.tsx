'use client';

import { useLanguage } from '@/contexts/LanguageContext';
import { useEffect } from 'react';

interface BilingualLayoutProps {
  children: React.ReactNode;
}

export function BilingualLayout({ children }: BilingualLayoutProps) {
  const { language } = useLanguage();

  useEffect(() => {
    // Update document language attribute and title
    if (typeof window !== 'undefined') {
      // Update the lang attribute on the html element
      const htmlElement = document.documentElement;
      if (htmlElement) {
        htmlElement.lang = language || 'vi';
      }

      // Update document title based on language
      const titles = {
        vi: "Cá Vàng - Đánh giá Nhận thức",
        en: "Cá Vàng - Cognitive Assessment System"
      };
      document.title = titles[language || 'vi'] || titles.vi;
    }
  }, [language]);

  return <>{children}</>;
}
