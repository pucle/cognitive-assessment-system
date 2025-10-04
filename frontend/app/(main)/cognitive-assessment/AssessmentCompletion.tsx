"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, CheckCircle, Mail, FileText, TrendingUp } from "lucide-react";

interface AssessmentCompletionProps {
  sessionId: string;
  mode: 'personal' | 'community';
  onComplete: () => void;
}

interface CompletionResult {
  redirect: string;
  data?: any;
  message?: string;
}

export const AssessmentCompletion: React.FC<AssessmentCompletionProps> = ({
  sessionId,
  mode,
  onComplete
}) => {
  const [isCompleting, setIsCompleting] = useState(false);
  const [completionResult, setCompletionResult] = useState<CompletionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleCompleteAssessment = async () => {
    setIsCompleting(true);
    setError(null);

    try {
      const response = await fetch('/api/assessment/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sessionId,
          mode
        })
      });

      const result = await response.json();

      if (result.success) {
        setCompletionResult(result.result);

        // Call completion callback
        onComplete();

        // Auto-redirect after a short delay for user to see the success message
        setTimeout(() => {
          if (mode === 'personal') {
            // For personal mode, navigate to results page (data will be fetched by the page)
            router.push(result.result.redirect);
          } else {
            router.push(result.result.redirect);
          }
        }, 3000);

      } else {
        setError(result.error || 'Hoàn thành đánh giá thất bại');
      }

    } catch (err) {
      console.error('Assessment completion error:', err);
      setError('Lỗi kết nối khi hoàn thành đánh giá');
    } finally {
      setIsCompleting(false);
    }
  };

  if (completionResult) {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="text-center flex items-center justify-center gap-2 text-green-600">
            <CheckCircle className="w-6 h-6" />
            Đánh giá hoàn thành thành công!
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          {mode === 'personal' ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <FileText className="w-8 h-8 mx-auto text-blue-500 mb-2" />
                  <div className="font-semibold">Báo cáo chi tiết</div>
                  <div className="text-sm text-gray-600">Với biểu đồ và phân tích</div>
                </div>

                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <TrendingUp className="w-8 h-8 mx-auto text-green-500 mb-2" />
                  <div className="font-semibold">Bài tập cá nhân hóa</div>
                  <div className="text-sm text-gray-600">Dựa trên kết quả của bạn</div>
                </div>

                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <Mail className="w-8 h-8 mx-auto text-purple-500 mb-2" />
                  <div className="font-semibold">Email báo cáo</div>
                  <div className="text-sm text-gray-600">Gửi kèm file đính kèm</div>
                </div>
              </div>

              <div className="text-center text-gray-600">
                <p>Đang chuyển hướng đến trang kết quả...</p>
                <p className="text-sm mt-1">Kết quả chi tiết và gợi ý bài tập đang chờ bạn!</p>
              </div>
            </div>
          ) : (
            <div className="text-center space-y-4">
              <div className="p-6 bg-green-50 rounded-lg">
                <Mail className="w-12 h-12 mx-auto text-green-500 mb-4" />
                <div className="font-semibold text-lg mb-2">Kết quả đã được gửi!</div>
                <div className="text-gray-600">
                  {completionResult.message || 'Kết quả đánh giá đã được gửi đến email của bạn'}
                </div>
              </div>

              <div className="text-center text-gray-600">
                <p>Đang chuyển hướng...</p>
                <p className="text-sm mt-1">Cảm ơn bạn đã tham gia đánh giá cộng đồng!</p>
              </div>
            </div>
          )}

          <div className="text-center">
        <Button
          onClick={() => router.push('/')}
          variant="ghost"
          className="mt-4"
        >
          Về trang chủ
        </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle className="text-center">
          Hoàn thành đánh giá
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {mode === 'personal' && (
          <div className="text-center space-y-2">
            <h3 className="font-semibold text-lg">Chế độ Cá nhân</h3>
            <p className="text-sm text-gray-600">
              Bạn sẽ nhận được báo cáo chi tiết với biểu đồ, phân tích chuyên sâu,
              và gợi ý bài tập cá nhân hóa qua email.
            </p>
          </div>
        )}

        {mode === 'community' && (
          <div className="text-center space-y-2">
            <h3 className="font-semibold text-lg">Chế độ Cộng đồng</h3>
            <p className="text-sm text-gray-600">
              Kết quả sẽ được gửi trực tiếp đến email của bạn
              với thông tin cơ bản về sức khỏe trí nhớ.
            </p>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}

        <Button
          onClick={handleCompleteAssessment}
          disabled={isCompleting}
          className="w-full"
          size="lg"
        >
          {isCompleting ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Đang xử lý...
            </>
          ) : (
            <>
              <CheckCircle className="w-4 h-4 mr-2" />
              Hoàn thành đánh giá
            </>
          )}
        </Button>

        <div className="text-xs text-gray-500 text-center">
          {isCompleting ? 'Vui lòng đợi trong giây lát...' : 'Nhấn để hoàn thành và nhận kết quả'}
        </div>
      </CardContent>
    </Card>
  );
};
