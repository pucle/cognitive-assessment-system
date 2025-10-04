"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { useSafeUser } from "@/app/hooks/useSafeClerk";
import { Loader2, X, Fish, Shell, Waves } from "lucide-react";

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

interface ProfilePopupProps {
  isOpen: boolean;
  onClose: () => void;
  onComplete: () => void;
}

export default function ProfilePopup({ isOpen, onClose, onComplete }: ProfilePopupProps) {
  // Use safe Clerk hook that handles availability
  const { user, isLoaded, isClerkAvailable } = useSafeUser();
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
  const [isNewUser, setIsNewUser] = useState(false);

  useEffect(() => {
    if (isLoaded && user && isOpen) {
      // Pre-fill email and image from Clerk if available
      setUserData(prev => ({
        ...prev,
        email: user.primaryEmailAddress?.emailAddress || "",
        name: user.fullName || "",
        imageSrc: user.imageUrl || ""
      }));

      // Check if user already has profile data
      fetchUserProfile();
    }
  }, [isLoaded, user, isOpen]);

  const fetchUserProfile = async () => {
    try {
      const res = await fetch("/api/profile");
      if (res.ok) {
        const data = await res.json();
        setUserData(data);
        setIsNewUser(false);
      } else {
        // Nếu không có profile, đây là user mới
        setIsNewUser(true);
      }
    } catch (error) {
      console.error("Error fetching profile:", error);
      setIsNewUser(true);
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
          onComplete(); // Call the completion callback
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

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Underwater background with fish decorations */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-400 via-cyan-300 to-teal-300 rounded-3xl">
              <div className="absolute top-4 right-8 text-white/30">
                <Fish className="w-8 h-8 rotate-12" />
              </div>
              <div className="absolute bottom-8 left-6 text-white/20">
                <Shell className="w-6 h-6" />
              </div>
              <div className="absolute top-1/2 right-4 text-white/25">
                <Waves className="w-5 h-5" />
              </div>
            </div>

            <Card className="relative p-8 backdrop-blur-xl bg-white/90 shadow-2xl rounded-3xl border border-white/50">
              {/* Close button */}
              <button
                onClick={onClose}
                className="absolute top-4 right-4 z-10 p-2 rounded-full bg-white/80 hover:bg-white transition-colors"
              >
                <X className="w-5 h-5 text-gray-600" />
              </button>

              <div className="text-center mb-8">
                <div className="flex justify-center mb-4">
                  <div className="relative">
                    <Fish className="w-12 h-12 text-blue-500 animate-bounce" />
                    <div className="absolute -top-2 -right-2 w-4 h-4 bg-yellow-400 rounded-full animate-pulse"></div>
                  </div>
                </div>
                <h2 className="text-3xl font-extrabold text-gray-800 mb-4 drop-shadow-sm">
                  {isNewUser ? '🌊 Chào mừng bạn đến với Cá Vàng!' : '🐠 Cập nhật hồ sơ'}
                </h2>
                <p className="text-gray-600 text-lg">
                  {isNewUser
                    ? 'Hãy cho chúng tôi biết thêm về bạn để cá nhân hóa trải nghiệm dưới nước'
                    : 'Cập nhật thông tin cá nhân của bạn'
                  }
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                    <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
                      {field.label} {field.required && <span className="text-red-500">*</span>}
                    </label>
                    {field.type === "select" ? (
                      <select
                        name={field.name}
                        value={userData[field.name as keyof UserData]}
                        onChange={handleChange}
                        className="px-4 py-3 rounded-xl border border-gray-300 bg-white/80 focus:outline-none focus:ring-2 focus:ring-blue-400 shadow-sm"
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
                        className={`px-4 py-3 rounded-xl border border-gray-300 bg-white/80 focus:outline-none focus:ring-2 focus:ring-blue-400 shadow-sm ${
                          field.disabled ? 'opacity-50 cursor-not-allowed' : ''
                        }`}
                        required={field.required}
                      />
                    )}
                  </div>
                ))}
              </div>

              {/* Thông báo về việc gửi email tự động */}
              <div className="mt-8 p-4 bg-blue-50 rounded-xl border border-blue-200">
                <div className="flex items-start gap-3">
                  <div className="text-blue-500 mt-0.5">📧</div>
                  <div>
                    <p className="text-sm font-medium text-blue-800">
                      Email xác nhận sẽ được gửi tự động
                    </p>
                    <p className="text-xs text-blue-600 mt-1">
                      Chúng tôi sẽ gửi bản sao thông tin hồ sơ của bạn qua email sau khi lưu thành công
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-center gap-4 mt-10">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button
                    onClick={handleSave}
                    className="px-8 py-3 rounded-xl bg-gradient-to-r from-blue-500 via-cyan-400 to-teal-500 text-white text-lg font-bold shadow-lg hover:shadow-xl transition-all duration-300"
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
        </motion.div>
      )}
    </AnimatePresence>
  );
}
