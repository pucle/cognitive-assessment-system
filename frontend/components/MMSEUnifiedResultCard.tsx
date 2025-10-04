"use client";

import React from 'react';
import { CheckCircle, Clock, AlertCircle, Brain, Mic, MessageSquare, Activity, Target, Zap } from 'lucide-react';

interface GPTEvaluation {
  vocabulary_score?: number;
  context_relevance_score: number;
  overall_score: number;
  analysis: string;
  feedback: string;
  vocabulary_analysis?: {
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
  };
  context_analysis: {
    relevance_level: 'high' | 'medium' | 'low';
    accuracy: 'accurate' | 'partially_accurate' | 'inaccurate';
    completeness: 'complete' | 'partial' | 'incomplete';
    issues: string[];
  };
  cognitive_assessment: {
    language_fluency: 'excellent' | 'good' | 'fair' | 'poor';
    cognitive_level: 'high' | 'medium' | 'low';
    attention_focus: 'good' | 'fair' | 'poor';
    memory_recall: 'excellent' | 'good' | 'fair' | 'poor';
  };
  transcript_info: {
    word_count: number;
    is_short_transcript: boolean;
    vocabulary_richness_applicable: boolean;
  };
}

interface AudioAnalysis {
  fluency: number; // 0-5 scale
  pronunciation: number; // 0-5 scale
  clarity: number; // 0-5 scale
  responseTime: number; // seconds
  pauseAnalysis: {
    averagePause: number; // seconds
    hesitationCount: number;
    cognitiveLoad: 'low' | 'medium' | 'high';
    description: string;
  };
  prosody: number; // 0-5 scale
  overallConfidence: number; // 0-100
}

interface ClinicalFeedback {
  overallAssessment: string;
  observations: string[];
  improvements: string[];
  confidence: number; // 0-100
}

interface QuestionResult {
  questionId: number;
  questionText: string;
  domain: string;
  transcript: string;
  transcriptionConfidence?: number;
  status: 'completed' | 'processing' | 'failed';
  gptEvaluation?: GPTEvaluation;
  audioAnalysis?: AudioAnalysis;
  clinicalFeedback?: ClinicalFeedback;
  processed_at: string;
}

interface MMSEUnifiedResultCardProps {
  questionResult: QuestionResult;
  isMMSEComplete: boolean;
  sessionId: string;
}

export function MMSEUnifiedResultCard({
  questionResult,
  isMMSEComplete,
  sessionId
}: MMSEUnifiedResultCardProps) {

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-blue-500 animate-pulse" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Brain className="w-5 h-5 text-gray-500" />;
    }
  };

  const getFluencyDescription = (score: number) => {
    if (score >= 4.5) return "Xuất sắc - lưu loát, tự nhiên";
    if (score >= 3.5) return "Tốt - mạch lạc, ít ngập ngừng";
    if (score >= 2.5) return "Khá - có chút ngập ngừng";
    if (score >= 1.5) return "Cần cải thiện - nhiều ngập ngừng";
    return "Yếu - rất ngập ngừng, khó theo dõi";
  };

  const getPronunciationDescription = (score: number) => {
    if (score >= 4.5) return "Xuất sắc - phát âm chuẩn xác";
    if (score >= 3.5) return "Tốt - phát âm rõ ràng";
    if (score >= 2.5) return "Khá - phát âm có thể chấp nhận";
    if (score >= 1.5) return "Cần cải thiện - phát âm không rõ";
    return "Yếu - phát âm khó hiểu";
  };

  const getClarityDescription = (score: number) => {
    if (score >= 4.5) return "Xuất sắc - âm thanh rất rõ";
    if (score >= 3.5) return "Tốt - âm thanh rõ ràng";
    if (score >= 2.5) return "Khá - âm thanh chấp nhận được";
    if (score >= 1.5) return "Cần cải thiện - âm thanh không rõ";
    return "Yếu - âm thanh kém chất lượng";
  };

  const getProsodyDescription = (score: number) => {
    if (score >= 4.5) return "Xuất sắc - ngữ điệu tự nhiên";
    if (score >= 3.5) return "Tốt - ngữ điệu phù hợp";
    if (score >= 2.5) return "Khá - ngữ điệu cơ bản";
    if (score >= 1.5) return "Cần cải thiện - ngữ điệu hạn chế";
    return "Yếu - ngữ điệu nghèo nàn";
  };

  const navigateToFinalResults = () => {
    window.location.href = `/results?sessionId=${sessionId}`;
  };

  return (
    <div className="mmse-result-card bg-[#F6E6DB] border border-[#EFD5C2] shadow-sm hover:shadow-md transition-shadow rounded-lg p-6">
      {/* Question Header */}
      <div className="question-header flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-amber-800 mb-1">
            Câu hỏi {questionResult.questionId}
          </h3>
          <p className="text-sm text-amber-700 mb-2">{questionResult.questionText}</p>
          <div className="flex items-center gap-2 text-xs text-amber-600">
            <span className="px-2 py-1 bg-amber-50 text-amber-700 rounded-full">
              Lĩnh vực: {questionResult.domain}
            </span>
            <span className="px-2 py-1 bg-amber-50 text-amber-700 rounded-full">
              Xử lý: {new Date(questionResult.processed_at).toLocaleString('vi-VN')}
            </span>
          </div>
        </div>
        <div className="ml-4">
          {getStatusIcon(questionResult.status)}
        </div>
      </div>

      {/* User Response */}
      <div className="user-response mb-6">
        <div className="flex items-center gap-2 mb-2">
          <Mic className="w-4 h-4 text-amber-600" />
          <h4 className="text-sm font-medium text-amber-800">Phản hồi của bạn:</h4>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 text-gray-800 text-sm leading-relaxed">
          {questionResult.transcript || "Không có transcript"}
        </div>
        {questionResult.transcriptionConfidence && (
          <div className="mt-2 text-xs text-gray-500">
            Độ tin cậy chuyển đổi: {Number(questionResult.transcriptionConfidence).toFixed(2)}%
          </div>
        )}
      </div>

      {/* GPT Clinical Evaluation */}
      {questionResult.gptEvaluation && (
        <div className="gpt-evaluation mb-6">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare className="w-4 h-4 text-amber-600" />
            <h4 className="text-sm font-medium text-amber-800">Đánh giá AI Lâm sàng:</h4>
          </div>

          <div className="space-y-3">
            {/* Overall Score */}
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-blue-700">Điểm tổng thể:</span>
                <span className="text-lg font-bold text-blue-600">
                  {typeof questionResult.gptEvaluation.overall_score === 'number' ? Number(questionResult.gptEvaluation.overall_score).toFixed(2) : 'N/A'}/10
                </span>
              </div>

              {/* Context Relevance */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-blue-600">Độ liên quan nội dung:</span>
                <span className="font-medium">
                  {questionResult.gptEvaluation.context_relevance_score ? Number(questionResult.gptEvaluation.context_relevance_score).toFixed(2) : 'N/A'}/10
                </span>
              </div>

              {/* Vocabulary Score (if applicable) */}
              {typeof questionResult.gptEvaluation.vocabulary_score === 'number' && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-blue-600">Độ phong phú từ vựng:</span>
                  <span className="font-medium">
                    {Number(questionResult.gptEvaluation.vocabulary_score).toFixed(2)}/10
                  </span>
                </div>
              )}
            </div>

            {/* Clinical Analysis */}
            <div className="bg-white border border-blue-200 rounded-lg p-3">
              <h5 className="text-sm font-medium text-gray-700 mb-2">Phân tích lâm sàng:</h5>
              <p className="text-sm text-gray-600 leading-relaxed">
                {questionResult.gptEvaluation.analysis || 'Không có phân tích'}
              </p>
            </div>

            {/* Cognitive Assessment */}
            {questionResult.gptEvaluation.cognitive_assessment && (
              <div className="grid grid-cols-2 gap-2">
                <div className="text-center p-2 bg-green-50 rounded">
                  <div className="text-xs text-green-600 font-medium">Trôi chảy ngôn ngữ</div>
                  <div className="text-sm font-bold text-green-700">
                    {questionResult.gptEvaluation.cognitive_assessment.language_fluency || 'N/A'}
                  </div>
                </div>
                <div className="text-center p-2 bg-blue-50 rounded">
                  <div className="text-xs text-blue-600 font-medium">Mức độ nhận thức</div>
                  <div className="text-sm font-bold text-blue-700">
                    {questionResult.gptEvaluation.cognitive_assessment.cognitive_level || 'N/A'}
                  </div>
                </div>
                <div className="text-center p-2 bg-purple-50 rounded">
                  <div className="text-xs text-purple-600 font-medium">Tập trung chú ý</div>
                  <div className="text-sm font-bold text-purple-700">
                    {questionResult.gptEvaluation.cognitive_assessment.attention_focus || 'N/A'}
                  </div>
                </div>
                <div className="text-center p-2 bg-orange-50 rounded">
                  <div className="text-xs text-orange-600 font-medium">Ghi nhớ</div>
                  <div className="text-sm font-bold text-orange-700">
                    {questionResult.gptEvaluation.cognitive_assessment.memory_recall || 'N/A'}
                  </div>
                </div>
              </div>
            )}

            {/* Improvement Suggestions */}
            {questionResult.gptEvaluation.feedback && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                <h5 className="text-sm font-medium text-amber-700 mb-2">Gợi ý cải thiện:</h5>
                <p className="text-sm text-amber-600 leading-relaxed">
                  {questionResult.gptEvaluation.feedback}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Audio Linguistics Analysis */}
      {questionResult.audioAnalysis && (
        <div className="audio-analysis mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-amber-600" />
            <h4 className="text-sm font-medium text-amber-800">Phân tích Ngôn ngữ Học Âm thanh:</h4>
          </div>

          <div className="space-y-3">
            {/* Audio Quality Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-green-50 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-green-600 mb-1">
                  {typeof questionResult.audioAnalysis.fluency === 'number' ? Number(questionResult.audioAnalysis.fluency).toFixed(2) : 'N/A'}/5
                </div>
                <div className="text-xs font-medium text-green-700">Lưu loát</div>
                <div className="text-xs text-green-600 mt-1">
                  {typeof questionResult.audioAnalysis.fluency === 'number' ? getFluencyDescription(questionResult.audioAnalysis.fluency) : 'Không có dữ liệu'}
                </div>
              </div>

              <div className="bg-blue-50 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-blue-600 mb-1">
                  {typeof questionResult.audioAnalysis.pronunciation === 'number' ? Number(questionResult.audioAnalysis.pronunciation).toFixed(2) : 'N/A'}/5
                </div>
                <div className="text-xs font-medium text-blue-700">Phát âm</div>
                <div className="text-xs text-blue-600 mt-1">
                  {typeof questionResult.audioAnalysis.pronunciation === 'number' ? getPronunciationDescription(questionResult.audioAnalysis.pronunciation) : 'Không có dữ liệu'}
                </div>
              </div>

              <div className="bg-purple-50 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-purple-600 mb-1">
                  {typeof questionResult.audioAnalysis.clarity === 'number' ? Number(questionResult.audioAnalysis.clarity).toFixed(2) : 'N/A'}/5
                </div>
                <div className="text-xs font-medium text-purple-700">Rõ ràng</div>
                <div className="text-xs text-purple-600 mt-1">
                  {typeof questionResult.audioAnalysis.clarity === 'number' ? getClarityDescription(questionResult.audioAnalysis.clarity) : 'Không có dữ liệu'}
                </div>
              </div>

              <div className="bg-orange-50 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-orange-600 mb-1">
                  {typeof questionResult.audioAnalysis.prosody === 'number' ? Number(questionResult.audioAnalysis.prosody).toFixed(2) : 'N/A'}/5
                </div>
                <div className="text-xs font-medium text-orange-700">Ngữ điệu</div>
                <div className="text-xs text-orange-600 mt-1">
                  {typeof questionResult.audioAnalysis.prosody === 'number' ? getProsodyDescription(questionResult.audioAnalysis.prosody) : 'Không có dữ liệu'}
                </div>
              </div>
            </div>

            {/* Response Time */}
            {questionResult.audioAnalysis.responseTime && (
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Thời gian phản hồi:</span>
                  <span className="text-sm font-bold text-gray-600">
                    {questionResult.audioAnalysis.responseTime.toFixed(2)} giây
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  Chỉ số tốc độ xử lý nhận thức - thời gian suy nghĩ trước khi trả lời
                </div>
              </div>
            )}

            {/* Pause Analysis */}
            {questionResult.audioAnalysis.pauseAnalysis && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <h5 className="text-sm font-medium text-red-700 mb-2">Phân tích khoảng dừng:</h5>
                <div className="grid grid-cols-3 gap-2 mb-2">
                  <div className="text-center">
                    <div className="text-sm font-bold text-red-600">
                      {questionResult.audioAnalysis.pauseAnalysis.averagePause?.toFixed(2) || 'N/A'}s
                    </div>
                    <div className="text-xs text-red-600">TB khoảng dừng</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-bold text-red-600">
                      {questionResult.audioAnalysis.pauseAnalysis.hesitationCount || 'N/A'}
                    </div>
                    <div className="text-xs text-red-600">Lần ngập ngừng</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-bold text-red-600 capitalize">
                      {questionResult.audioAnalysis.pauseAnalysis.cognitiveLoad || 'N/A'}
                    </div>
                    <div className="text-xs text-red-600">Tải nhận thức</div>
                  </div>
                </div>
                <p className="text-xs text-red-600 leading-relaxed">
                  {questionResult.audioAnalysis.pauseAnalysis.description || 'Không có mô tả'}
                </p>
              </div>
            )}

            {/* Overall Audio Confidence */}
            {typeof questionResult.audioAnalysis.overallConfidence === 'number' && (
              <div className="bg-green-50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-green-700">Độ tin cậy âm thanh tổng thể:</span>
                  <span className="text-sm font-bold text-green-600">
                    {Number(questionResult.audioAnalysis.overallConfidence).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-green-200 rounded-full h-2">
                  <div
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${questionResult.audioAnalysis.overallConfidence}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Clinical Feedback Integration */}
      {questionResult.clinicalFeedback && (
        <div className="clinical-feedback mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Target className="w-4 h-4 text-amber-600" />
            <h4 className="text-sm font-medium text-amber-800">Đánh giá Lâm sàng Tổng hợp:</h4>
          </div>

          <div className="space-y-3">
            {/* Overall Assessment */}
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
              <h5 className="text-sm font-medium text-purple-700 mb-2">Đánh giá tổng thể:</h5>
              <p className="text-sm text-purple-600 leading-relaxed">
                {questionResult.clinicalFeedback.overallAssessment || 'Không có đánh giá tổng thể'}
              </p>
            </div>

            {/* Clinical Observations */}
            {questionResult.clinicalFeedback.observations && questionResult.clinicalFeedback.observations.length > 0 && (
              <div className="bg-blue-50 rounded-lg p-3">
                <h5 className="text-sm font-medium text-blue-700 mb-2">Quan sát lâm sàng:</h5>
                <ul className="text-sm text-blue-600 space-y-1">
                  {questionResult.clinicalFeedback.observations.map((obs, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>{obs}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Improvement Suggestions */}
            {questionResult.clinicalFeedback.improvements && questionResult.clinicalFeedback.improvements.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                <h5 className="text-sm font-medium text-amber-700 mb-2">Đề xuất cải thiện:</h5>
                <ul className="text-sm text-amber-600 space-y-1">
                  {questionResult.clinicalFeedback.improvements.map((imp, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-amber-500 mt-1">•</span>
                      <span>{imp}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Combined Confidence */}
            {typeof questionResult.clinicalFeedback.confidence === 'number' && (
              <div className="bg-indigo-50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-indigo-700">Độ tin cậy tổng hợp:</span>
                  <span className="text-sm font-bold text-indigo-600">
                    {Number(questionResult.clinicalFeedback.confidence).toFixed(2)}%
                  </span>
                </div>
                <div className="text-xs text-indigo-600">
                  Kết hợp đánh giá GPT, phân tích âm thanh và quan sát lâm sàng
                </div>
                <div className="w-full bg-indigo-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${questionResult.clinicalFeedback.confidence}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
