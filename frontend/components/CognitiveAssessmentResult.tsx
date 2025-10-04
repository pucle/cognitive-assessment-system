"use client";

import { Brain, TrendingUp, AlertTriangle, CheckCircle, FileText } from 'lucide-react';
import { motion } from 'framer-motion';

interface AssessmentResult {
  success: boolean;
  error?: string;
  mmse_prediction?: {
    predicted_mmse: number;
    severity: string;
    description: string;
    confidence: number;
  };
  transcript?: string;
  method_used?: string;
  metadata?: Record<string, unknown>;
}

interface CognitiveAssessmentResultProps {
  result: AssessmentResult;
  onReset?: () => void;
}

export function CognitiveAssessmentResult({ result, onReset }: CognitiveAssessmentResultProps) {
  if (!result || !result.success) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-red-700 mb-2">Đánh giá thất bại</h3>
        <p className="text-red-600">{result?.error || 'Không thể xử lý yêu cầu'}</p>
      </div>
    );
  }

  const { mmse_prediction, transcript, method_used, metadata } = result;
  const {
    predicted_mmse = 0,
    severity = 'Không xác định',
    description = 'Không có mô tả',
    confidence = 0
  } = (mmse_prediction && Object.keys(mmse_prediction).length > 0) ? mmse_prediction : {
    predicted_mmse: 0,
    severity: 'Không xác định',
    description: 'Không có mô tả',
    confidence: 0
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'bình thường':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'suy giảm nhẹ':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'suy giảm trung bình':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'suy giảm nặng':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'bình thường':
        return <CheckCircle className="w-5 h-5" />;
      case 'suy giảm nhẹ':
        return <TrendingUp className="w-5 h-5" />;
      case 'suy giảm trung bình':
        return <AlertTriangle className="w-5 h-5" />;
      case 'suy giảm nặng':
        return <AlertTriangle className="w-5 h-5" />;
      default:
        return <Brain className="w-5 h-5" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="text-center">
        <Brain className="w-16 h-16 text-blue-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Kết quả Đánh giá Nhận thức</h2>
        <p className="text-gray-600">Dựa trên ML model với độ chính xác cao</p>
      </div>

      {/* Assessment Status - No Individual MMSE Scores */}
      <div className="bg-gradient-to-r from-green-500 to-blue-600 rounded-2xl p-6 text-white text-center">
        <CheckCircle className="w-12 h-12 mx-auto mb-4" />
        <div className="text-lg font-bold mb-2">Đánh giá hoàn tất</div>
        <div className="text-sm opacity-90">Điểm số MMSE đầy đủ có sẵn tại trang kết quả</div>
      </div>

      {/* Severity Assessment */}
      <div className={`border-2 rounded-xl p-4 ${getSeverityColor(severity)}`}>
        <div className="flex items-center gap-3 mb-2">
          {getSeverityIcon(severity)}
          <h3 className="text-lg font-semibold">Mức độ: {severity}</h3>
        </div>
        <p className="text-sm">{description}</p>
      </div>

      {/* Confidence */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="font-medium text-gray-700">Độ tin cậy:</span>
          <span className="text-lg font-semibold text-blue-600">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${confidence * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Transcript */}
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <FileText className="w-5 h-5 text-gray-600" />
          <h3 className="font-medium text-gray-700">Transcript:</h3>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 text-gray-800 text-sm">
          {transcript || 'Không có transcript'}
        </div>
        <div className="text-xs text-gray-500 mt-2">
          Phương thức: {method_used === 'manual_input' ? 'Nhập thủ công' : 'Ghi âm'}
        </div>
      </div>

      {/* Metadata */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
        <h3 className="font-medium text-gray-700 mb-3">Thông tin chi tiết:</h3>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-500">Câu hỏi:</span>
            <p className="font-medium">{(metadata?.question && typeof metadata.question === 'string' && metadata.question.trim().length > 0) ? metadata.question : 'N/A'}</p>
          </div>
          <div>
            <span className="text-gray-500">Tuổi:</span>
            <p className="font-medium">{(metadata?.age && typeof metadata.age === 'string' && metadata.age.trim().length > 0) ? metadata.age : 'N/A'}</p>
          </div>
          <div>
            <span className="text-gray-500">Giới tính:</span>
            <p className="font-medium">{(metadata?.gender && typeof metadata.gender === 'string' && metadata.gender.trim().length > 0) ? metadata.gender : 'N/A'}</p>
          </div>
          <div>
            <span className="text-gray-500">Thời gian:</span>
            <p className="font-medium">
              {metadata?.timestamp && typeof metadata.timestamp === 'string' ? new Date(metadata.timestamp).toLocaleString('vi-VN') : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      {onReset && (
        <div className="flex gap-3">
          <button
            onClick={onReset}
            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-medium transition-colors"
          >
            Đánh giá mới
          </button>
          <button
            onClick={() => window.print()}
            className="px-6 py-3 border border-gray-300 rounded-xl font-medium hover:bg-gray-50 transition-colors"
          >
            In kết quả
          </button>
        </div>
      )}

      {/* Disclaimer */}
      <div className="text-xs text-gray-500 text-center border-t pt-4">
        <p>⚠️ Kết quả này chỉ mang tính chất tham khảo và không thay thế cho chẩn đoán y tế chuyên nghiệp.</p>
        <p>Vui lòng tham khảo ý kiến bác sĩ nếu có vấn đề về sức khỏe.</p>
      </div>
    </motion.div>
  );
}
