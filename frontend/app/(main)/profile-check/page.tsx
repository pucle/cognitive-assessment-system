// app/(main)/profile-check/page.tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useSafeUser } from "@/app/hooks/useSafeClerk";
import { Loader2 } from "lucide-react";

export default function ProfileCheckPage() {
  const { user, isLoaded } = useSafeUser();
  const router = useRouter();

  useEffect(() => {
    if (!isLoaded) return;

    const checkProfile = async () => {
      try {
        const res = await fetch("/api/profile");
        if (res.ok) {
          const data = await res.json();
          // Nếu có đầy đủ thông tin cơ bản, chuyển đến menu
          if (data.name && data.age && data.gender && data.phone) {
            router.push("/menu");
          } else {
            // Nếu chưa có thông tin, chuyển đến trang profile
            router.push("/user-profile");
          }
        } else {
          // Nếu chưa có profile (lỗi 404), chuyển đến trang profile
          router.push("/user-profile");
        }
      } catch (error) {
        console.error("Error checking profile:", error);
        // Nếu có lỗi, an toàn là chuyển đến trang profile
        router.push("/user-profile");
      }
    };

    if (user) {
      checkProfile();
    }
  }, [isLoaded, user, router]);

  return (
    <div className="flex justify-center items-center min-h-screen">
      <div className="text-center">
        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
        <p className="text-gray-600">Đang kiểm tra thông tin...</p>
      </div>
    </div>
  );
}