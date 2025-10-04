// app/(main)/user-profile/page.tsx
"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
// import { useUser } from "@clerk/nextjs"; // Temporarily disabled
import { Loader2, Menu, ArrowLeft } from "lucide-react";
import Link from "next/link";

interface UserData {
  name: string;
  age: string;
  gender: string;
  email: string;
  phone: string;
  title?: string;
  imageSrc?: string;
  mmseScore?: number;
}

export default function ProfilePage() {
  // const { user, isLoaded } = useUser(); // Temporarily disabled
  const user = null; // Mock user
  const isLoaded = true; // Mock loaded state
  const router = useRouter();
  const [userData, setUserData] = useState<UserData>({
    name: "",
    age: "",
    gender: "",
    email: "",
    phone: "",
    title: "",
    imageSrc: "",
    mmseScore: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [isNewUser, setIsNewUser] = useState(false);

  useEffect(() => {
    // Always fetch user profile on component mount
    // Use a default email for testing/demo purposes
    const defaultEmail = "ledinhphuc1408@gmail.com"; // You can change this for testing

    setUserData(prev => ({
      ...prev,
      email: defaultEmail
    }));

    fetchUserProfile(defaultEmail);
  }, []);

  const fetchUserProfile = async (email: string) => {
    try {
      const res = await fetch(`/api/profile?email=${encodeURIComponent(email)}`);
      if (res.ok) {
        const response = await res.json();
        if (response.success && response.data) {
          setUserData(response.data);
          setIsNewUser(false);
        } else {
          // If no profile data found, this is a new user
          setIsNewUser(true);
        }
      } else {
        // If API returns error (like 404 user not found), this is a new user
        setIsNewUser(true);
      }
    } catch (error) {
      console.error("Error fetching profile:", error);
      setIsNewUser(true);
    } finally {
      setIsInitialLoad(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setUserData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSave = async () => {
    if (!userData.name || !userData.age || !userData.gender || !userData.phone) {
      alert("Vui lòng điền đầy đủ thông tin!");
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch("/api/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(userData),
      });
      
      if (res.ok) {
        const result = await res.json();
        if (result.success) {
          // Tự động gửi email mà không cần hỏi user
          try {
            await fetch("/api/send-email", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ userData }),
            });
            alert("Cập nhật thành công! Email xác nhận đã được gửi đến bạn.");
          } catch (emailError) {
            console.error("Email error:", emailError);
            alert("Cập nhật thành công! Tuy nhiên có lỗi khi gửi email xác nhận.");
          }
          router.push("/menu");
        } else {
          alert("Có lỗi xảy ra: " + (result.error || "Unknown"));
        }
      } else {
        const err = await res.json();
        alert("Có lỗi xảy ra: " + (err.error || "Unknown"));
      }
    } catch (error) {
      alert("Có lỗi xảy ra khi lưu thông tin!");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSkip = () => {
    router.push("/menu");
  };

  if (!isLoaded || isInitialLoad) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p style={{ color: '#8B6D57' }}>Đang tải thông tin...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)'
    }}>
      {/* Header with hamburger menu */}
      <div className="sticky top-0 z-50 backdrop-blur-sm p-2" style={{
        background: 'rgba(255, 255, 255, 0.9)',
        borderBottom: '2px solid #F4A261'
      }}>
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-2">
            <div className="md:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <Menu className="h-5 w-5" style={{ color: '#F4A261' }} />
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
                <ArrowLeft className="h-5 w-5" style={{ color: '#F4A261' }} />
              </Button>
            </Link>
          </div>
          <h1 className="font-bold text-lg" style={{ color: '#B8763E' }}>
            Hồ sơ cá nhân
          </h1>
          <div />
        </div>
      </div>

      <div className="flex justify-center p-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-3xl"
        >
        <Card className="p-8 rounded-3xl" style={{
          background: 'rgba(255, 255, 255, 0.9)',
          border: '2px solid #F4A261',
          boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
        }}>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-extrabold mb-4 drop-shadow-sm" style={{ color: '#B8763E' }}>
              {isNewUser ? '👋 Chào mừng bạn đến với Cá Vàng!' : '✏️ Cập nhật hồ sơ'}
            </h2>
            <p className="text-lg" style={{ color: '#8B6D57' }}>
              {isNewUser 
                ? 'Hãy cho chúng tôi biết thêm về bạn để cá nhân hóa trải nghiệm' 
                : 'Cập nhật thông tin cá nhân của bạn'
              }
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              { 
                label: "Họ và tên", 
                name: "name", 
                type: "text", 
                placeholder: "Nhập họ và tên",
                required: true
              },
              { 
                label: "Tuổi", 
                name: "age", 
                type: "number", 
                placeholder: "Nhập tuổi",
                required: true
              },
              {
                label: "Giới tính",
                name: "gender",
                type: "select",
                options: ["Nam", "Nữ", "Khác"],
                required: true
              },
              { 
                label: "Email", 
                name: "email", 
                type: "email", 
                placeholder: "Nhập email",
                disabled: true
              },
              { 
                label: "Số điện thoại", 
                name: "phone", 
                type: "text", 
                placeholder: "Nhập số điện thoại",
                required: true
              },
            ].map((field, idx) => (
              <div key={idx} className="flex flex-col gap-2">
                <label className="text-sm font-medium" style={{ color: '#B8763E' }}>
                  {field.label} {field.required && <span className="text-red-500">*</span>}
                </label>
                {field.type === "select" ? (
                  <select
                    name={field.name}
                    value={userData[field.name as keyof UserData]}
                    onChange={handleChange}
                    className="px-4 py-3 rounded-xl border bg-white/80 focus:outline-none focus:ring-2 shadow-sm" style={{
                      border: '2px solid #F4A261',
                      backgroundColor: 'rgba(255, 255, 255, 0.9)'
                    }}
                    required={field.required}
                  >
                    <option value="">Chọn giới tính</option>
                    {field.options?.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type={field.type}
                    name={field.name}
                    value={userData[field.name as keyof UserData]}
                    onChange={handleChange}
                    placeholder={field.placeholder}
                    disabled={field.disabled}
                    className={`px-4 py-3 rounded-xl bg-white/80 focus:outline-none focus:ring-2 shadow-sm ${
                      field.disabled ? 'opacity-50 cursor-not-allowed' : ''
                    }`} style={{
                      border: '2px solid #F4A261',
                      backgroundColor: 'rgba(255, 255, 255, 0.9)'
                    }}
                    required={field.required}
                  />
                )}
              </div>
            ))}
          </div>

          {/* Thông báo về việc gửi email tự động */}
          <div className="mt-8 p-4 rounded-xl" style={{
            background: 'rgba(244, 162, 97, 0.1)',
            border: '2px solid #F4A261'
          }}>
            <div className="flex items-start gap-3">
              <div style={{ color: '#F4A261' }} className="mt-0.5">📧</div>
              <div>
                <p className="text-sm font-medium" style={{ color: '#B8763E' }}>
                  Email xác nhận sẽ được gửi tự động
                </p>
                <p className="text-xs mt-1" style={{ color: '#8B6D57' }}>
                  Chúng tôi sẽ gửi bản sao thông tin hồ sơ của bạn qua email sau khi lưu thành công
                </p>
              </div>
            </div>
          </div>

          <div className="flex justify-center gap-4 mt-10">
            {!isNewUser && (
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button
                  onClick={handleSkip}
                  variant="primaryOutline"
                  className="px-6 py-3 rounded-xl text-lg font-medium" style={{
                    border: '2px solid #F4A261',
                    color: '#B8763E',
                    backgroundColor: 'rgba(255, 255, 255, 0.9)'
                  }}
                  disabled={isLoading}
                >
                  Bỏ qua
                </Button>
              </motion.div>
            )}
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                onClick={handleSave}
                className="px-8 py-3 rounded-xl text-white text-lg font-bold shadow-lg transition-all duration-300" style={{
                  background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)'
                }}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Đang lưu...
                  </>
                ) : (
                  <>Lưu thông tin</>
                )}
              </Button>
            </motion.div>
          </div>
        </Card>
      </motion.div>
      </div>
    </div>
  );
}