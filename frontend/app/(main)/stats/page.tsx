"use client";

import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";
import { useLanguage } from "@/contexts/LanguageContext";
import { Menu, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";
import { MMSETrendChart } from "@/components/MMSETrendChart";
import { useUser } from "@clerk/nextjs";
import MmseLineChart from "@/components/charts/MmseLineChart";
import DetailedResultCard from "@/components/results/DetailedResultCard";

interface PersonalTestResult {
  id: number;
  sessionId: string;
  userEmail: string;
  userName: string;
  questionId: string;
  questionText: string;
  autoTranscript: string;
  manualTranscript: string;
  createdAt: string;
}

interface PatientAssessment {
  id: number;
  sessionId: string;
  name: string;
  email: string;
  age: string;
  gender: string;
  phone: string;
  status: string;
  finalMmse: number | null;
  overallGptScore: number | null;
  resultsJson: string;
  createdAt: string;
  updatedAt: string;
}

interface CognitiveRow {
  id: number;
  sessionId: string;
  userInfo: any;
  completedAt: string;
  finalMmseScore: number;
  overallGptScore: number;
  questionResults: any[];
  cognitiveAnalysis?: any;
  status: string;
  totalQuestions: number;
  answeredQuestions: number;
  completionRate: number;
  createdAt: string;
}

export default function StatsPage() {
  const { t } = useLanguage();

  let clerkUser = null;
  let isLoaded = false;

  try {
    const clerkData = useUser();
    clerkUser = clerkData.user;
    isLoaded = clerkData.isLoaded;
  } catch (error) {
    console.warn('Clerk not available, using fallback:', error);
  }

  const [viewMode, setViewMode] = useState<'personal' | 'community'>('personal');
  const [userEmail, setUserEmail] = useState<string>('');
  const [currentUserId, setCurrentUserId] = useState<string>('');
  const [cognitiveResults, setCognitiveResults] = useState<CognitiveRow[]>([]);
  const [trainingData, setTrainingData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isLoaded) {
      if (clerkUser) {
        const email = clerkUser.primaryEmailAddress?.emailAddress ||
                     clerkUser.emailAddresses?.[0]?.emailAddress;
        const userId = clerkUser.id;

        setUserEmail(email || '');
        setCurrentUserId(userId || email || 'demo_user');
        console.log('✅ User authenticated:', { userId, email });
      } else {
        const fallbackInfo = getFallbackUserInfo();
        setCurrentUserId(fallbackInfo.userId);
        setUserEmail(fallbackInfo.email);
        console.log('⚠️ Using fallback user info');
      }
    }
  }, [clerkUser, isLoaded]);

  useEffect(() => {
    if (currentUserId && isLoaded) {
      console.log('🔄 Stats page useEffect triggered, fetching data...');
      fetchAllData();
    }
  }, [viewMode, currentUserId, userEmail, isLoaded]);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    console.log(`📊 Starting data fetch for ${viewMode} mode, userId: ${currentUserId}`);

    try {
      await fetchCognitiveResults();
      await fetchTrainingData();
    } catch (err) {
      console.error('❌ Error fetching stats data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  const fetchCognitiveResults = async () => {
    console.log('🔍 Fetching cognitive assessment results...');

    const apiUrl = '/api/get-cognitive-assessment-results';
    const params = new URLSearchParams();

    if (viewMode === 'personal') {
      params.append('userId', currentUserId);
      params.append('usageMode', 'personal');
    } else {
      params.append('usageMode', 'community');
      params.append('userEmail', userEmail);
    }

    const fullUrl = `${apiUrl}?${params.toString()}`;
    console.log('📡 Calling API:', fullUrl);

    const response = await fetch(fullUrl);
    const result = await response.json();

    console.log('📥 API Response:', {
      success: result.success,
      count: result.count || result.data?.length || 0,
      status: response.status
    });

    if (result.success) {
      const transformedData = (result.data || []).map((item: any): CognitiveRow => ({
        id: item.id,
        sessionId: item.sessionId,
        userInfo: item.userInfo,
        completedAt: item.completedAt || item.createdAt,
        finalMmseScore: item.finalMmseScore ?? 0,
        overallGptScore: item.overallGptScore ?? 0,
        questionResults: item.questionResults || [],
        cognitiveAnalysis: item.cognitiveAnalysis,
        status: item.status || 'completed',
        totalQuestions: item.totalQuestions || 12,
        answeredQuestions: item.answeredQuestions || 12,
        completionRate: item.completionRate || 100,
        createdAt: item.createdAt,
      }));
      setCognitiveResults(transformedData);
      console.log(`✅ Loaded ${transformedData.length} cognitive assessment results`);
    } else {
      throw new Error(result.error || 'Failed to fetch cognitive results');
    }
  };

  const fetchTrainingData = async () => {
    console.log('🔍 Fetching training samples...');

    try {
      const apiUrl = '/api/get-training-samples';
      const params = new URLSearchParams();

      if (viewMode === 'personal') {
        params.append('userId', currentUserId);
        params.append('usageMode', 'personal');
      } else {
        params.append('usageMode', 'community');
        params.append('userEmail', userEmail);
      }

      const fullUrl = `${apiUrl}?${params.toString()}`;
      console.log('📡 Calling Training API:', fullUrl);

      const response = await fetch(fullUrl);
      const result = await response.json();

      console.log('📥 Training API Response:', {
        success: result.success,
        count: result.count || result.data?.length || 0,
        status: response.status
      });

      if (result.success) {
        setTrainingData(result.data || []);
        console.log(`✅ Loaded ${result.data?.length || 0} training samples`);
      } else {
        console.warn('⚠️ Training data fetch failed:', result.error);
        setTrainingData([]);
      }
    } catch (err) {
      console.warn('⚠️ Training data fetch error:', err);
      setTrainingData([]);
    }
  };

  const handleModeChange = (mode: 'personal' | 'community') => {
    console.log(`🔄 Switching to ${mode} mode`);
    setViewMode(mode);
  };

  const handleRefresh = () => {
    console.log('🔄 Manual refresh triggered');
    fetchAllData();
  };

  const handleViewDetails = (result: CognitiveRow) => {
    console.log('View details for:', result);
    window.location.href = `/results/${encodeURIComponent(result.sessionId)}`;
  };

  const getFallbackUserInfo = () => {
    if (typeof window !== 'undefined') {
      const userInfo = localStorage.getItem('userInfo');
      if (userInfo) {
        try {
          return JSON.parse(userInfo);
        } catch {
          // Ignore invalid JSON
        }
      }
    }

    return {
      userId: 'demo_user',
      email: 'demo@example.com'
    };
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('vi-VN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getScoreColor = (score: number) => {
    return "text-amber-900";
  };

  const getScoreBadgeColor = (score: number) => {
    if (score >= 80) return "bg-green-600";
    if (score >= 60) return "bg-yellow-600";
    if (score >= 40) return "bg-orange-600";
    return "bg-red-600";
  };

  const totalAssessments = cognitiveResults.length;
  const averageMmseScore = totalAssessments > 0
    ? (cognitiveResults.reduce((sum, item) => sum + (item.finalMmseScore || 0), 0) / totalAssessments).toFixed(2)
    : '0';
  const averageGptScore = totalAssessments > 0
    ? (cognitiveResults.reduce((sum, item) => sum + (item.overallGptScore || 0), 0) / totalAssessments).toFixed(2)
    : '0';

  return (
    <div className="min-h-screen" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif',
      color: '#8B6D57'
    }}>
      <header className="border-b-2" style={{ 
        borderColor: '#F4A261',
        background: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)'
      }}>
        <div className="max-w-7xl mx-auto px-8 py-6">
          <h1 className="text-3xl font-bold mb-2" style={{ color: '#B8763E' }}>
            Tổng quan Thống kê
          </h1>
          <p className="text-lg" style={{ color: '#8B6D57' }}>
            Tiến độ học tập và thành tích của bạn
          </p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-8 py-10">
        <div className="flex justify-between items-center mb-10">
          <div></div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 shadow-lg"
            style={{
              background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
              color: '#FFFFFF',
              border: '2px solid #E67635'
            }}
          >
            {loading ? 'Đang làm mới...' : 'Làm mới Dữ liệu'}
          </button>
        </div>

        <div className="mb-10">
          <div className="flex gap-6">
            <button
              onClick={() => handleModeChange('personal')}
              disabled={loading}
              className="px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 shadow-lg"
              style={{
                background: viewMode === 'personal' ? 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)' : 'rgba(255, 255, 255, 0.9)',
                color: viewMode === 'personal' ? '#FFFFFF' : '#B8763E',
                border: `2px solid ${viewMode === 'personal' ? '#E67635' : '#F4A261'}`,
                boxShadow: viewMode === 'personal' ? '0 8px 16px rgba(244, 162, 97, 0.4)' : 'none'
              }}
            >
              Chế độ Cá nhân
            </button>
            <button
              onClick={() => handleModeChange('community')}
              disabled={loading}
              className="px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 shadow-lg"
              style={{
                background: viewMode === 'community' ? 'linear-gradient(135deg, #E88D4D 0%, #E67635 100%)' : 'rgba(255, 255, 255, 0.9)',
                color: viewMode === 'community' ? '#FFFFFF' : '#B8763E',
                border: `2px solid ${viewMode === 'community' ? '#D96B2F' : '#E88D4D'}`,
                boxShadow: viewMode === 'community' ? '0 8px 16px rgba(232, 141, 77, 0.4)' : 'none'
              }}
            >
              Chế độ Cộng đồng
            </button>
          </div>
        </div>

        <div className="mb-10 p-8 rounded-3xl shadow-lg" style={{
          background: 'rgba(255, 255, 255, 0.9)',
          borderLeft: `6px solid ${viewMode === 'personal' ? '#F4A261' : '#E88D4D'}`,
          border: '2px solid #FAE6D0'
        }}>
          {viewMode === 'personal' ? (
            <p className="text-xl font-bold" style={{ color: '#8B6D57' }}>
              <span style={{ color: '#F4A261', fontWeight: '700' }}>Chế độ Cá nhân:</span> Hiển thị kết quả của bạn (ID: {currentUserId})
            </p>
          ) : (
            <p className="text-xl font-bold" style={{ color: '#8B6D57' }}>
              <span style={{ color: '#E88D4D', fontWeight: '700' }}>Chế độ Cộng đồng:</span> Hiển thị kết quả cộng đồng cho email: {userEmail}
            </p>
          )}
        </div>

        {loading && (
          <div className="text-center py-20">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-4" style={{
              borderTopColor: '#F4A261',
              borderRightColor: '#E88D4D',
              borderBottomColor: '#E67635',
              borderLeftColor: 'transparent'
            }}></div>
            <p className="mt-6 text-2xl font-bold" style={{ color: '#B8763E' }}>Đang tải dữ liệu của bạn...</p>
          </div>
        )}

        {error && (
          <div className="mb-10 p-8 rounded-3xl shadow-lg" style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #DC2626',
            borderLeft: '6px solid #DC2626'
          }}>
            <p className="text-xl font-bold mb-6" style={{ color: '#8B6D57' }}>
              <span style={{ color: '#DC2626' }}>Lỗi:</span> {error}
            </p>
            <button
              onClick={handleRefresh}
              className="px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 shadow-lg"
              style={{
                background: 'linear-gradient(135deg, #DC2626 0%, #B91C1C 100%)',
                color: '#FFFFFF',
                border: '2px solid #991B1B'
              }}
            >
              Thử lại
            </button>
          </div>
        )}

        {!loading && !error && (
          <div className="space-y-10">
            {viewMode === 'personal' && (
              <MmseLineChart data={cognitiveResults} />
            )}

            <motion.div
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div
                className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
                style={{
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '2px solid #F4A261',
                  boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
                }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold" style={{ color: '#B8763E' }}>Tổng số Đánh giá</h3>
                </div>
                <div className="space-y-3">
                  <div className="text-5xl font-bold" style={{ color: '#F4A261' }}>{totalAssessments}</div>
                  <div className="text-lg" style={{ color: '#8B6D57' }}>Đánh giá đã hoàn thành</div>
                </div>
              </div>

              <div
                className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
                style={{
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '2px solid #E88D4D',
                  boxShadow: '0 8px 16px rgba(232, 141, 77, 0.2)'
                }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold" style={{ color: '#B8763E' }}>Điểm MMSE Trung bình</h3>
                </div>
                <div className="space-y-3">
                  <div className="text-5xl font-bold" style={{ color: '#E88D4D' }}>{averageMmseScore}</div>
                  <div className="text-lg" style={{ color: '#8B6D57' }}>Điểm nhận thức</div>
                </div>
              </div>

              <div
                className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
                style={{
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '2px solid #E67635',
                  boxShadow: '0 8px 16px rgba(230, 118, 53, 0.2)'
                }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold" style={{ color: '#B8763E' }}>Điểm GPT Trung bình</h3>
                </div>
                <div className="space-y-3">
                  <div className="text-5xl font-bold" style={{ color: '#E67635' }}>{averageGptScore}</div>
                  <div className="text-lg" style={{ color: '#8B6D57' }}>Điểm đánh giá AI</div>
                </div>
              </div>

              <div
                className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
                style={{
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '2px solid #D96B2F',
                  boxShadow: '0 8px 16px rgba(217, 107, 47, 0.2)'
                }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold" style={{ color: '#B8763E' }}>Mẫu Huấn luyện</h3>
                </div>
                <div className="space-y-3">
                  <div className="text-5xl font-bold" style={{ color: '#D96B2F' }}>{trainingData.length}</div>
                  <div className="text-lg" style={{ color: '#8B6D57' }}>Dữ liệu huấn luyện AI</div>
                </div>
              </div>
            </motion.div>

            {totalAssessments > 0 && (
              <div>
                <div className="mb-10">
                  <h2 className="text-3xl font-bold mb-3" style={{ color: '#B8763E' }}>
                    Chi tiết Đánh giá
                  </h2>
                  <p className="text-xl" style={{ color: '#8B6D57' }}>
                    Xem kết quả chi tiết cho mỗi đánh giá với phân tích AI toàn diện
                  </p>
                </div>

                <div className="space-y-8">
                  {cognitiveResults.map((result, index) => (
                    <div key={result.id || index}>
                      <DetailedResultCard
                        result={result}
                        onViewDetails={handleViewDetails}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {totalAssessments === 0 && (
              <motion.div
                className="text-center py-20 rounded-3xl shadow-lg"
                style={{
                  background: 'rgba(255, 255, 255, 0.9)',
                  border: '2px solid #F4A261'
                }}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="mb-10">
                  <div
                    className="w-24 h-24 mx-auto rounded-full flex items-center justify-center mb-8 shadow-lg"
                    style={{ background: 'linear-gradient(135deg, #FAE6D0 0%, #F5D7BE 100%)' }}
                  >
                    <div className="w-12 h-12 rounded-full shadow-lg" style={{ background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)' }}></div>
                  </div>
                </div>
                <h3 className="text-3xl font-bold mb-6" style={{ color: '#B8763E' }}>
                  {viewMode === 'personal' ? 'Chưa có Đánh giá nào' : 'Không có Dữ liệu Cộng đồng'}
                </h3>
                <p className="text-xl mb-10 max-w-2xl mx-auto leading-relaxed" style={{ color: '#8B6D57' }}>
                  {viewMode === 'personal'
                    ? 'Bạn chưa hoàn thành bất kỳ đánh giá nhận thức nào. Bắt đầu đánh giá đầu tiên để xem biểu đồ tiến độ và phân tích chi tiết.'
                    : 'Không có dữ liệu cộng đồng nào khả dụng cho địa chỉ email này. Tham gia đánh giá cộng đồng để đóng góp và xem kết quả.'
                  }
                </p>
                <div className="flex flex-col sm:flex-row gap-6 justify-center">
                  <button
                    onClick={() => window.location.href = '/cognitive-assessment'}
                    className="px-10 py-5 rounded-2xl font-bold text-xl transition-all duration-300 hover:scale-105 shadow-lg"
                    style={{
                      background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                      color: '#FFFFFF',
                      border: '2px solid #E67635'
                    }}
                  >
                    Bắt đầu Đánh giá
                  </button>
                  <button
                    onClick={() => window.location.href = '/menu'}
                    className="px-10 py-5 rounded-2xl font-bold text-xl transition-all duration-300 hover:scale-105 shadow-lg"
                    style={{
                      background: 'rgba(255, 255, 255, 0.9)',
                      color: '#B8763E',
                      border: '2px solid #F4A261'
                    }}
                  >
                    Xem Menu
                  </button>
                </div>
              </motion.div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatsContent({
  data,
  mode,
  formatDate
}: {
  data: CognitiveRow[];
  mode: string;
  formatDate: (date: string) => string;
}) {
  const sortedResults = [...data].sort((a, b) =>
    new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
  );

  const chartData = sortedResults.map((result) => ({
    date: new Date(result.createdAt).toLocaleDateString('vi-VN', {
      month: 'short',
      day: 'numeric'
    }),
    mmse: result.finalMmseScore || 0,
    session: result.sessionId
  }));

  return (
    <div className="space-y-10" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif'
    }}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div
          className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261',
            boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
          }}
        >
          <h3 className="text-xl font-bold mb-6" style={{ color: '#B8763E' }}>Tổng số Đánh giá</h3>
          <p className="text-4xl font-bold" style={{ color: '#F4A261' }}>{data.length}</p>
        </div>
        <div
          className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #E88D4D',
            boxShadow: '0 8px 16px rgba(232, 141, 77, 0.2)'
          }}
        >
          <h3 className="text-xl font-bold mb-6" style={{ color: '#B8763E' }}>Điểm MMSE Trung bình</h3>
          <p className="text-4xl font-bold" style={{ color: '#E88D4D' }}>
            {data.length > 0
              ? (data.reduce((sum, item) => sum + (item.finalMmseScore || 0), 0) / data.length).toFixed(2)
              : 'N/A'
            }
          </p>
        </div>
        <div
          className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #E67635',
            boxShadow: '0 8px 16px rgba(230, 118, 53, 0.2)'
          }}
        >
          <h3 className="text-xl font-bold mb-6" style={{ color: '#B8763E' }}>Điểm GPT Trung bình</h3>
          <p className="text-4xl font-bold" style={{ color: '#E67635' }}>
            {data.length > 0
              ? (data.reduce((sum, item) => sum + (item.overallGptScore || 0), 0) / data.length).toFixed(2)
              : 'N/A'
            }
          </p>
        </div>
      </div>

      <h3 className="text-2xl font-bold" style={{ color: '#B8763E' }}>Lịch sử Điểm MMSE</h3>

      {chartData.length > 1 && (
        <div
          className="p-8 rounded-3xl shadow-lg border-2"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261',
            boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
          }}
        >
          <h4 className="text-xl font-bold mb-8" style={{ color: '#B8763E' }}>Xu hướng Điểm MMSE Theo Thời gian</h4>
          <div className="h-64">
            <MMSETrendChart data={chartData} />
          </div>
        </div>
      )}

      <h3 className="text-2xl font-bold" style={{ color: '#B8763E' }}>
        {mode === 'personal' ? 'Đánh giá Cá nhân Gần đây' : 'Đánh giá Cộng đồng Gần đây'}
      </h3>

      {data.length === 0 ? (
        <div
          className="p-8 rounded-3xl text-center shadow-lg"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261',
            color: '#8B6D57'
          }}
        >
          Không có kết quả nào khả dụng.
        </div>
      ) : (
        <div className="space-y-6">
          {data.slice(0, 10).map((row) => (
            <motion.div
              key={`${row.sessionId}-${row.createdAt}-${row.id || 'no-id'}`}
              className="p-8 rounded-3xl shadow-lg border-2 transition-all duration-300 hover:scale-105"
              style={{
                background: 'rgba(255, 255, 255, 0.9)',
                border: '2px solid #F4A261',
                boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
              }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-6 mb-4">
                    <div className="text-xl font-bold" style={{ color: '#B8763E' }}>Phiên: {row.sessionId}</div>
                    {mode === 'community' && row.userInfo?.name && (
                      <div className="text-lg px-4 py-2 rounded-full font-bold shadow-lg" style={{
                        background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                        color: '#FFFFFF'
                      }}>
                        {row.userInfo.name}
                      </div>
                    )}
                  </div>
                  <div className="text-lg mb-3" style={{ color: '#8B6D57' }}>{formatDate(row.createdAt)}</div>
                  {mode === 'community' && row.userInfo?.email && (
                    <div className="text-lg" style={{ color: '#B8763E' }}>{row.userInfo.email}</div>
                  )}
                </div>
                <div className="flex items-center gap-8">
                  <div className="text-center">
                    <div className="text-lg font-bold mb-2" style={{ color: '#8B6D57' }}>Điểm MMSE</div>
                    <div className="text-4xl font-bold" style={{ color: '#E88D4D' }}>{row.finalMmseScore ?? 'N/A'}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold mb-2" style={{ color: '#8B6D57' }}>Điểm GPT</div>
                    <div className="text-4xl font-bold" style={{ color: '#E67635' }}>{row.overallGptScore ?? 'N/A'}</div>
                  </div>
                  <div className="ml-auto">
                    <Link href={`/results/${encodeURIComponent(row.sessionId)}`}>
                      <button className="px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 shadow-lg"
                        style={{
                          background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                          color: '#FFFFFF',
                          border: '2px solid #E67635'
                        }}>
                        Xem Chi tiết
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}

function PersonalStatsView({
  results,
  formatDate
}: {
  results: PersonalTestResult[];
  formatDate: (date: string) => string;
}) {
  return (
    <div className="space-y-10" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif'
    }}>
      <h3 className="text-2xl font-bold" style={{ color: '#B8763E' }}>Kết quả Đánh giá Nhận thức Hoàn chỉnh</h3>
      {results.length === 0 ? (
        <motion.div
          className="p-10 text-center rounded-3xl shadow-lg"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261'
          }}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h3 className="text-3xl font-bold mb-6" style={{ color: '#B8763E' }}>
            Chưa hoàn thành bài kiểm tra nào
          </h3>
          <p className="text-xl mb-8" style={{ color: '#8B6D57' }}>
            Bắt đầu với bài kiểm tra MMSE để xem thống kê của bạn
          </p>
          <Link href="/cognitive-assessment">
            <button className="px-10 py-5 rounded-2xl font-bold text-xl transition-all duration-300 hover:scale-105 shadow-lg"
              style={{
                background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                color: '#FFFFFF',
                border: '2px solid #E67635'
              }}>
              Bắt đầu Đánh giá
            </button>
          </Link>
        </motion.div>
      ) : (
        results.map((result) => (
          <motion.div
            key={`${result.sessionId}-${result.createdAt}-${result.id || 'personal'}`}
            className="p-8 rounded-3xl shadow-lg border-2"
            style={{
              background: 'rgba(255, 255, 255, 0.9)',
              border: '2px solid #F4A261',
              boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-6">
                <div>
                  <h3 className="text-2xl font-bold" style={{ color: '#B8763E' }}>
                    Phiên: {result.sessionId}
                  </h3>
                  <p className="text-lg mt-2" style={{ color: '#8B6D57' }}>
                    {formatDate(result.createdAt)}
                  </p>
                  <p className="text-lg" style={{ color: '#B8763E' }}>
                    Người dùng: {result.userName || result.userEmail || 'N/A'}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="px-6 py-3 rounded-full text-lg font-bold shadow-lg"
                  style={{
                    background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                    color: '#FFFFFF'
                  }}>
                  Đánh giá Hoàn chỉnh
                </div>
                <Link href={`/results/${encodeURIComponent(result.sessionId)}`}>
                  <button className="px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-300 hover:scale-105 shadow-lg"
                    style={{
                      background: 'linear-gradient(135deg, #E88D4D 0%, #E67635 100%)',
                      color: '#FFFFFF',
                      border: '2px solid #D96B2F'
                    }}>
                    View Details
                  </button>
                </Link>
              </div>
            </div>

            <div className="space-y-6">
              <div className="p-6 rounded-2xl border-2"
                style={{
                  background: 'rgba(254, 243, 226, 0.5)',
                  border: '2px solid #FAE6D0'
                }}>
                <div className="mb-4">
                  <h4 className="font-bold text-xl" style={{ color: '#B8763E' }}>
                    {result.questionText}
                  </h4>
                </div>

                <div className="text-lg space-y-4">
                  <div>
                    <span className="font-bold" style={{ color: '#B8763E' }}>Kết quả Đánh giá:</span>
                    <p className="mt-2" style={{ color: '#8B6D57' }}>
                      {result.autoTranscript}
                    </p>
                  </div>
                  {result.manualTranscript && result.manualTranscript !== '{}' && (
                    <div>
                      <span className="font-bold" style={{ color: '#B8763E' }}>Chi tiết Câu hỏi:</span>
                      <div className="mt-2 p-4 rounded-xl text-base"
                        style={{
                          background: 'rgba(255, 255, 255, 0.9)',
                          color: '#8B6D57',
                          maxHeight: '10rem',
                          overflowY: 'auto',
                          border: '2px solid #F5D7BE'
                        }}>
                        {result.manualTranscript.length > 200
                          ? `${result.manualTranscript.substring(0, 200)}...`
                          : result.manualTranscript
                        }
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        ))
      )}
    </div>
  );
}

function PatientStatsView({
  results,
  formatDate,
  getScoreColor,
  getScoreBadgeColor
}: {
  results: PatientAssessment[];
  formatDate: (date: string) => string;
  getScoreColor: (score: number) => string;
  getScoreBadgeColor: (score: number) => string;
}) {
  return (
    <div className="space-y-10" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif'
    }}>
      {results.length === 0 ? (
        <motion.div
          className="p-10 text-center rounded-3xl shadow-lg"
          style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261'
          }}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h3 className="text-3xl font-bold mb-6" style={{ color: '#B8763E' }}>
            Chưa có Bệnh nhân nào
          </h3>
          <p className="text-xl" style={{ color: '#8B6D57' }}>
            Đánh giá bệnh nhân sẽ xuất hiện ở đây
          </p>
        </motion.div>
      ) : (
        results.map((patient) => (
          <motion.div
            key={`${patient.sessionId}-${patient.createdAt}-${patient.id || 'patient'}`}
            className="p-8 rounded-3xl shadow-lg border-2"
            style={{
              background: 'rgba(255, 255, 255, 0.9)',
              border: '2px solid #F4A261',
              boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-6">
                <div>
                  <h3 className="text-2xl font-bold" style={{ color: '#B8763E' }}>
                    {patient.name}
                  </h3>
                  <p className="text-lg mt-2" style={{ color: '#8B6D57' }}>
                    {patient.age} years • {patient.gender}
                  </p>
                  <p className="text-lg" style={{ color: '#B8763E' }}>{patient.email}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="mb-4 px-6 py-3 rounded-full text-lg font-bold inline-block shadow-lg"
                  style={{
                    background: patient.status === 'completed' 
                      ? 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)' 
                      : 'linear-gradient(135deg, #E88D4D 0%, #E67635 100%)',
                    color: '#FFFFFF'
                  }}>
                  MMSE: {patient.finalMmse || 'N/A'}
                </div>
                <p className="text-lg" style={{ color: '#8B6D57' }}>
                  {formatDate(patient.createdAt)}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="text-center p-6 rounded-2xl border-2"
                style={{
                  background: 'rgba(254, 243, 226, 0.5)',
                  border: '2px solid #FAE6D0'
                }}>
                <div className="text-4xl font-bold mb-2" style={{ color: '#E88D4D' }}>
                  {patient.finalMmse || 'N/A'}
                </div>
                <div className="text-lg font-bold" style={{ color: '#8B6D57' }}>Điểm MMSE</div>
              </div>
              <div className="text-center p-6 rounded-2xl border-2"
                style={{
                  background: 'rgba(254, 243, 226, 0.5)',
                  border: '2px solid #FAE6D0'
                }}>
                <div className="text-4xl font-bold mb-2" style={{ color: '#E67635' }}>
                  {patient.overallGptScore || 'N/A'}
                </div>
                <div className="text-lg font-bold" style={{ color: '#8B6D57' }}>Điểm GPT</div>
              </div>
              <div className="text-center p-6 rounded-2xl border-2"
                style={{
                  background: 'rgba(254, 243, 226, 0.5)',
                  border: '2px solid #FAE6D0'
                }}>
                <div className="px-5 py-2 rounded-full text-lg font-bold inline-block shadow-lg"
                  style={{
                    background: patient.status === 'completed' 
                      ? 'linear-gradient(135deg, #10B981 0%, #059669 100%)' 
                      : 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                    color: '#FFFFFF'
                  }}>
                  {patient.status === 'completed' ? 'Hoàn thành' : 'Đang tiến hành'}
                </div>
                <div className="text-lg font-bold mt-3" style={{ color: '#8B6D57' }}>Trạng thái</div>
              </div>
            </div>

            {patient.resultsJson && (
              <div>
                <h4 className="font-bold text-xl mb-4" style={{ color: '#B8763E' }}>
                  Chi tiết Kết quả
                </h4>
                <div className="p-4 rounded-2xl text-lg max-h-40 overflow-y-auto border-2"
                  style={{
                    background: 'rgba(254, 243, 226, 0.5)',
                    border: '2px solid #FAE6D0',
                    color: '#8B6D57'
                  }}>
                  <pre className="whitespace-pre-wrap">
                    {patient.resultsJson}
                  </pre>
                </div>
              </div>
            )}
          </motion.div>
        ))
      )}
    </div>
  );
}