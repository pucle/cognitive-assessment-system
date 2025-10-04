"use client";

import React from 'react';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, Mail, Home, Users } from "lucide-react";

export default function CommunityThankYou() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-lg shadow-lg">
        <CardHeader className="text-center">
          <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <CardTitle className="text-2xl font-bold text-gray-800">
            Cảm ơn bạn đã tham gia!
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Success Message */}
          <div className="text-center space-y-3">
            <div className="flex items-center justify-center gap-2 text-green-600">
              <Mail className="w-5 h-5" />
              <span className="font-semibold">Kết quả đã được gửi!</span>
            </div>
            <p className="text-gray-600">
              Kết quả đánh giá sức khỏe trí nhớ của bạn đã được gửi đến email.
              Vui lòng kiểm tra hộp thư đến của bạn.
            </p>
          </div>

          {/* Community Stats Card */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4 text-blue-600" />
              <span className="font-semibold text-blue-800 text-sm">Chế độ Cộng đồng</span>
            </div>
            <p className="text-xs text-blue-700">
              Bạn đã góp phần vào việc thu thập dữ liệu sức khỏe cộng đồng.
              Cảm ơn sự đóng góp quý báu của bạn!
            </p>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <Button
              onClick={() => router.push('/')}
              className="w-full"
              size="lg"
            >
              <Home className="w-4 h-4 mr-2" />
              Về trang chủ
            </Button>

            <Button
              onClick={() => router.push('/settings')}
              variant="ghost"
              className="w-full"
            >
              <Users className="w-4 h-4 mr-2" />
              Đổi sang chế độ Cá nhân
            </Button>
          </div>

          {/* Footer Note */}
          <div className="text-center text-xs text-gray-500 pt-4 border-t">
            <p>
              Nếu bạn không nhận được email, vui lòng kiểm tra thư mục spam
              hoặc liên hệ với chúng tôi để được hỗ trợ.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
