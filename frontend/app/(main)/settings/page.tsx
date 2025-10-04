// app/(main)/settings/page.tsx
"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RoundedListItem } from "@/components/ui/rounded-list-item";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";
import { Wallet, LinkIcon, Globe, History, HelpCircle, Settings, LogOut, Menu, ArrowLeft } from "lucide-react";
import Link from "next/link";
// import { useClerk } from "@clerk/nextjs"; // Temporarily disabled
import { motion } from "framer-motion";
import { useLanguage } from "@/contexts/LanguageContext";
import { LanguageSwitcher } from "@/components/LanguageSwitcher";

export default function SettingsPage() {
  // const { signOut } = useClerk(); // Temporarily disabled
  const signOut = () => {
    // Mock sign out - just redirect to home
    window.location.href = '/';
  };
  const [user, setUser] = useState<Record<string, unknown> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { t, language, setLanguage } = useLanguage();
  const [trainingMode, setTrainingMode] = useState<boolean>(false);
  const [usageMode, setUsageMode] = useState<'personal' | 'community'>('personal');

  // Function to fetch user data from database
  const fetchUserDataFromDatabase = async () => {
    try {
      setIsLoading(true);
      
      // Get user ID or email from localStorage
      const storedUserData = typeof window !== 'undefined' ? localStorage.getItem('userData') : null;
      if (storedUserData) {
        const parsedData = JSON.parse(storedUserData);
        
        if (parsedData.id || parsedData.email) {
          // Try to get fresh data from database
          const response = await fetch(`/api/database/user?userId=${parsedData.id || ''}&email=${parsedData.email || ''}`);
          
          if (response.ok) {
            const result = await response.json();
            if (result.success && result.user) {
              const freshUserData = {
                id: result.user.id,
                name: result.user.name,
                age: result.user.age,
                gender: result.user.gender,
                email: result.user.email,
                phone: result.user.phone
              };
              
              setUser(freshUserData);
              // Update localStorage with fresh data
              if (typeof window !== 'undefined') {
                localStorage.setItem('userData', JSON.stringify(freshUserData));
              }
              console.log('Settings: User data updated from database');
              return;
            }
          }
        }
      }
      
      // Fallback to localStorage data
      if (storedUserData) {
        setUser(JSON.parse(storedUserData));
      }
    } catch (error) {
      console.error('Error fetching user data in settings:', error);
      // Fallback to localStorage
      const storedUserData = typeof window !== 'undefined' ? localStorage.getItem("userData") : null;
      if (storedUserData) {
        setUser(JSON.parse(storedUserData));
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUserDataFromDatabase();
  }, []);

  // Auto-refresh user data every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchUserDataFromDatabase();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // load/save settings
  useEffect(() => {
    try {
      const tm = typeof window !== 'undefined' ? localStorage.getItem('trainingMode') : null;
      const um = typeof window !== 'undefined' ? localStorage.getItem('usageMode') : null;
      if (tm !== null) setTrainingMode(tm === 'true');
      if (um === 'community' || um === 'personal') setUsageMode(um);
    } catch {}
  }, []);

  const toggleTrainingMode = () => {
    setTrainingMode(prev => {
      const next = !prev;
      try {
      if (typeof window !== 'undefined') {
        localStorage.setItem('trainingMode', String(next));
      }
    } catch {}
      return next;
    });
  };

  const setUsage = (mode: 'personal' | 'community') => {
    setUsageMode(mode);
    try {
      if (typeof window !== 'undefined') {
        localStorage.setItem('usageMode', mode);
      }
    } catch {}
  };

  const handleLanguageChange = (newLanguage: 'en' | 'vi') => {
    setLanguage(newLanguage);
  };

  return (
    <div className="bg-gradient-to-br from-amber-100 via-orange-100 to-rose-100 min-h-screen max-h-[150vh] overflow-auto">
      {/* Header with hamburger menu */}
      <div className="sticky top-0 z-50 bg-amber-100/95 backdrop-blur-sm border-b border-amber-300 p-2">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-2">
            <div className="md:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <Menu className="h-5 w-5 text-amber-700" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="p-0 w-80">
                  <SheetHeader>
                    <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
                  </SheetHeader>
                  <Sidebar />
                </SheetContent>
              </Sheet>
            </div>
            <Link href="/menu">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-5 w-5 text-amber-700" />
              </Button>
            </Link>
          </div>
          <h1 className="font-bold text-lg text-amber-800">
            Cài đặt
          </h1>
          <div />
        </div>
      </div>

      <div className="p-2 sm:p-3 lg:p-4 flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-3xl"
        >
        {/* Thông tin tài khoản */}
        <Card className="p-3 sm:p-4 lg:p-6 backdrop-blur-xl bg-white/70 shadow-2xl rounded-2xl sm:rounded-3xl border border-white/30">
          <div className="flex justify-between items-center mb-4 sm:mb-6 lg:mb-8">
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">
              ⚙️ {t('settings')}
            </h2>
            <Button
              onClick={fetchUserDataFromDatabase}
              disabled={isLoading}
              className="px-3 py-2 sm:px-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
              title="Refresh user data from database"
            >
              {isLoading ? '🔄' : '🔄'}
            </Button>
          </div>
          
          {isLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-gray-600">Đang tải thông tin từ database...</p>
            </div>
          ) : user ? (
            <div className="space-y-3">
              <RoundedListItem
                icon={<Wallet className="w-5 h-5 text-gray-700" />}
                title={(user.name as string) || t('full_name')}
                subtitle={(user.email as string) || ''}
              />
              <RoundedListItem title={t('age')} subtitle={(user.age as string) || '—'} />
              <RoundedListItem title={t('gender')} subtitle={(user.gender as string) || '—'} />
              <RoundedListItem title={t('phone')} subtitle={(user.phone as string) || '—'} />
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-600">Không tìm thấy thông tin người dùng</p>
            </div>
          )}
          
          <div className="flex justify-center mt-8">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link href="/user-profile">
                <Button className="px-8 py-3 rounded-xl bg-gradient-to-r from-blue-400 to-teal-500 text-white font-bold shadow-lg hover:shadow-xl transition">
                  ✏ {t('edit_profile')}
                </Button>
              </Link>
            </motion.div>
          </div>
        </Card>

        {/* Language & Actions list */}
        <Card className="p-3 sm:p-4 lg:p-5 mt-3 sm:mt-4 lg:mt-6 backdrop-blur-xl bg-white/70 shadow-xl rounded-2xl sm:rounded-3xl border border-white/30">
          <div className="space-y-2 sm:space-y-3">
            <div className="w-full flex items-center gap-2 sm:gap-3 px-2 sm:px-3 lg:px-4 py-2 sm:py-3 lg:py-3.5 bg-white/80 hover:bg-white rounded-xl sm:rounded-2xl border border-white/60 shadow-sm transition">
              <div className="shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-gray-100 flex items-center justify-center">
                <Globe className="w-4 h-4 sm:w-5 sm:h-5" />
              </div>
              <div className="flex-1 text-left">
                <div className="text-xs sm:text-sm font-semibold text-gray-800">{t('language_settings')}</div>
                <div className="text-xs text-gray-500">{language === 'vi' ? t('vietnamese') : t('english')}</div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex gap-2">
                  <Button size="sm" className="px-2 sm:px-3" onClick={() => handleLanguageChange('vi')}>VI</Button>
                  <Button size="sm" className="px-2 sm:px-3" onClick={() => handleLanguageChange('en')}>EN</Button>
                </div>
              </div>
            </div>
            {/* Usage mode */}
            <div className="w-full flex items-center gap-2 sm:gap-3 px-2 sm:px-3 lg:px-4 py-2 sm:py-3 lg:py-3.5 bg-white/80 hover:bg-white rounded-xl sm:rounded-2xl border border-white/60 shadow-sm transition">
              <div className="shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-amber-100 flex items-center justify-center text-sm">🏷️</div>
              <div className="flex-1 text-left">
                <div className="text-xs sm:text-sm font-semibold text-gray-800">Chế độ sử dụng</div>
                <div className="text-xs text-gray-500">{usageMode === 'community' ? 'Dịch vụ cộng đồng' : 'Cá nhân'}</div>
              </div>
              <div className="flex items-center gap-2">
                <Button size="sm" variant={usageMode==='personal' ? undefined : 'secondaryOutline'} onClick={() => setUsage('personal')}>Cá nhân</Button>
                <Button size="sm" variant={usageMode==='community' ? undefined : 'secondaryOutline'} onClick={() => setUsage('community')}>Cộng đồng</Button>
              </div>
            </div>
            {/* Training mode */}
            <div className="w-full flex items-center gap-3 px-3 py-3 sm:px-4 sm:py-3.5 bg-white/80 hover:bg-white rounded-2xl border border-white/60 shadow-sm transition">
              <div className="shrink-0 w-10 h-10 rounded-xl bg-amber-100 flex items-center justify-center">🎧</div>
              <div className="flex-1 text-left">
                <div className="text-sm font-semibold text-gray-800">Training Mode</div>
                <div className="text-xs text-gray-500">Ghi âm xong yêu cầu nhập transcript thủ công</div>
              </div>
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={toggleTrainingMode} className={trainingMode ? 'bg-amber-500 hover:bg-amber-600 text-white' : ''}>
                  {trainingMode ? 'Bật' : 'Tắt'}
                </Button>
              </div>
            </div>
            <RoundedListItem icon={<History className="w-5 h-5" />} title={t('check_history')} />
            <RoundedListItem icon={<HelpCircle className="w-5 h-5" />} title={t('help_center')} />
            <RoundedListItem icon={<Settings className="w-5 h-5" />} title={t('settings')} />
          </div>
        </Card>

        {/* Đăng xuất */}
        <Card className="p-8 mt-6 backdrop-blur-xl bg-white/70 shadow-xl rounded-3xl border border-white/30 text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">🚪 {t('logout')}</h2>
          <p className="mb-6 text-gray-700">{t('logout_confirmation')}</p>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              onClick={() => signOut()}
              className="px-8 py-3 rounded-xl bg-gradient-to-r from-red-500 to-pink-500 text-white font-bold shadow-lg hover:shadow-xl transition"
            >
              {t('logout')}
            </Button>
          </motion.div>
        </Card>
      </motion.div>
      </div>
    </div>
  );
}
