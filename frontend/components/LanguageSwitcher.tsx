'use client';

import { useLanguage } from '@/contexts/LanguageContext';
import { Button } from '@/components/ui/button';
import { Globe, Check } from 'lucide-react';

export function LanguageSwitcher() {
  const { language, setLanguage, t } = useLanguage();

  const handleLanguageChange = (newLanguage: 'vi' | 'en') => {
    setLanguage(newLanguage);
  };

  return (
    <div className="flex items-center gap-1">
      <Button
        variant={language === 'vi' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => handleLanguageChange('vi')}
        className={`flex items-center gap-1 px-2 py-1 text-xs ${
          language === 'vi'
            ? 'bg-blue-500 hover:bg-blue-600 text-white'
            : 'hover:bg-gray-100 text-gray-600'
        }`}
      >
        🇻🇳 VI
        {language === 'vi' && <Check className="h-3 w-3" />}
      </Button>
      <Button
        variant={language === 'en' ? 'default' : 'ghost'}
        size="sm"
        onClick={() => handleLanguageChange('en')}
        className={`flex items-center gap-1 px-2 py-1 text-xs ${
          language === 'en'
            ? 'bg-blue-500 hover:bg-blue-600 text-white'
            : 'hover:bg-gray-100 text-gray-600'
        }`}
      >
        🇺🇸 EN
        {language === 'en' && <Check className="h-3 w-3" />}
      </Button>
    </div>
  );
}
