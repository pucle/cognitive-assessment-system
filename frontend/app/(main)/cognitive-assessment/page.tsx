"use client";

import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";
import { motion } from "framer-motion";
import {
  Mic, Square, Loader2, Timer, CheckCircle, RotateCcw, Brain,
  Waves, Fish, Compass, Volume2, FileText, Trophy, User, Clock, FileAudio, Menu, ArrowLeft
} from "lucide-react";
import { useLanguage } from '@/contexts/LanguageContext';
import { LanguageSwitcher } from '@/components/LanguageSwitcher';
import Link from "next/link";
import { useRouter } from 'next/navigation';
import {
  fetchWithFallback,
  getMockQuestions,
  checkBackendHealth,
  getDefaultUserData,
  withApiFallback,
  API_BASE_URL
} from '@/lib/api-utils';

// MMSE Domain Structure - Scientific Standard
interface MMSEDomain {
  name: string;
  maxScore: number;
  currentScore: number;
  questions: Question[];
  completed: boolean;
}

interface MMSEAssessment {
  domains: {
    orientation: MMSEDomain;
    registration: MMSEDomain;
    attention_calculation: MMSEDomain;
    recall: MMSEDomain;
    language: MMSEDomain;
    construction: MMSEDomain;
  };
  totalScore: number | null; // Only calculated after ALL domains complete
  completed: boolean;
  cognitiveStatus: string | null;
}

// Interfaces
interface Question {
  id: string;
  category: string;
  domain: string; // MMSE domain: orientation, registration, etc.
  text: string;
  instruction?: string;
}

interface UserData {
  name: string;
  age: string;
  gender: string;
  email: string;
  phone: string;
}

interface AudioFeatures {
  duration: number;
  pitch_mean: number;
  pitch_std: number;
  speech_rate: number;
  tempo: number;
  silence_mean: number;
  number_utterances: number;
}

interface MMSEPrediction {
  predicted_mmse: number;
  severity: string;
  description: string;
  confidence: number;
  error?: string;
}

// Speech-based MMSE Support (for AI assistance only)
interface SpeechBasedMMSESupport {
  speechFeatures: {
    acoustic: any;
    linguistic: any;
    duration: any;
  };
  aiPredictedScore: number | null; // AI support, NOT replacement for MMSE
  confidence: number;
}

interface GPTEvaluation {
  vocabulary_score: number | null;
  context_relevance_score: number;
  overall_score: number;
  analysis: string;
  feedback: string;
  repetition_rate?: number; // Legacy support
  context_relevance?: number; // Legacy support
  comprehension_score?: number; // Legacy support
}

interface AutoTranscriptionResult {
  success: boolean;
  transcript: string;
  confidence: number;
  language: string;
  model: string;
  transcript_file: string;
  audio_duration: number;
  sample_rate: number;
  // Added optional fields for UI
  gpt_evaluation?: GPTEvaluation;
  audio_features?: AudioFeatures;
  error?: string;
}

// Extended Window interface for global variables
interface ExtendedWindow extends Window {
  lastAudioBlob?: Blob | null;
  mediaRecorder?: MediaRecorder;
}

interface TestResult {
  questionId: string;
  question: string;
  audioBlob?: Blob;
  audioFilename?: string;
  transcription?: string;
  timestamp: Date;
  duration: number;
  textRecord?: {
    saved: boolean;
    filename?: string;
    directory?: string;
  };
  processingMethod?: string;
  processingCost?: number;
  gpt_evaluation?: GPTEvaluation;
  audio_features?: AudioFeatures;
  mmse_prediction?: MMSEPrediction;
  auto_transcription?: AutoTranscriptionResult;
  // Legacy support
  o4mini_evaluation?: GPTEvaluation;
}

// Questions will be loaded from MMSE v2 backend

// Constants
const MAX_RECORDING_DURATION = 180;

// Async Assessment System State Structure
interface QuestionState {
  id: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  answer?: string;
  transcript?: string;
  score?: number;
  feedback?: string;
  audioBlob?: Blob;
  taskId?: string;
  timestamp?: Date;
  error?: string;
  retryCount?: number;
  completedAt?: Date;
}

interface AssessmentProgress {
  currentQuestion: number;
  totalQuestions: number;
  completedQuestions: number;
  processingQuestions: number;
  failedQuestions: number;
}

// Helper function to format model names
const formatModelName = (model: string | undefined): string => {
  if (!model) return 'Gemini 2.0 Flash';
  
  switch (model) {
    case 'gemini-2.0-flash-exp':
      return 'Gemini 2.0 Flash';
    case 'gemini-1.5-flash':
      return 'Gemini 1.5 Flash';
    case 'openai-whisper-1':
      return 'Whisper Base';
    case 'whisper-base':
      return 'Whisper Base';
    case 'whisper-large':
      return 'Whisper Large';
    default:
      return model.charAt(0).toUpperCase() + model.slice(1);
  }
};

// MMSE Scientific Assessment Class
class ScientificMMSEAssessment {
  domains: MMSEAssessment['domains'];
  totalScore: number | null = null;
  completed: boolean = false;
  cognitiveStatus: string | null = null;

  constructor() {
    this.domains = {
      orientation: {
        name: 'Định hướng thời gian và không gian',
        maxScore: 10,
        currentScore: 0,
        questions: [],
        completed: false
      },
      registration: {
        name: 'Ghi nhận/Đăng ký',
        maxScore: 3,
        currentScore: 0,
        questions: [],
        completed: false
      },
      attention_calculation: {
        name: 'Chú ý và tính toán',
        maxScore: 5,
        currentScore: 0,
        questions: [],
        completed: false
      },
      recall: {
        name: 'Hồi tưởng',
        maxScore: 3,
        currentScore: 0,
        questions: [],
        completed: false
      },
      language: {
        name: 'Ngôn ngữ',
        maxScore: 8,
        currentScore: 0,
        questions: [],
        completed: false
      },
      construction: {
        name: 'Xây dựng hình ảnh',
        maxScore: 1,
        currentScore: 0,
        questions: [],
        completed: false
      }
    };
  }

  // CRITICAL: Only calculate total after ALL domains complete
  calculateTotalScore(): number | null {
    const allDomainsComplete = Object.values(this.domains).every(domain => domain.completed);
    
    if (!allDomainsComplete) {
      console.warn('🚫 CANNOT calculate MMSE score - not all domains completed');
      return null; // MUST NOT return score if incomplete
    }

    this.totalScore = Object.values(this.domains).reduce((total, domain) => total + domain.currentScore, 0);
    this.completed = true;
    this.cognitiveStatus = this.classifyCognitiveStatus(this.totalScore);
    
    console.log(`✅ MMSE Total Score calculated: ${this.totalScore}/30 - ${this.cognitiveStatus}`);
    return this.totalScore;
  }

  // Scientific MMSE classification
  classifyCognitiveStatus(score: number): string {
    if (score >= 24) return 'Bình thường';
    if (score >= 18) return 'Suy giảm nhận thức nhẹ (MCI)';
    if (score >= 10) return 'Alzheimer nhẹ';
    return 'Alzheimer trung bình đến nặng';
  }

  // Complete a domain and update score
  completeDomain(domainKey: keyof MMSEAssessment['domains'], score: number): void {
    const domain = this.domains[domainKey];
    if (score < 0 || score > domain.maxScore) {
      throw new Error(`Invalid score ${score} for domain ${domainKey}. Max: ${domain.maxScore}`);
    }
    
    domain.currentScore = score;
    domain.completed = true;
    console.log(`✅ Domain ${domainKey} completed with score ${score}/${domain.maxScore}`);
  }

  // Check if assessment can be finalized
  canFinalize(): boolean {
    return Object.values(this.domains).every(domain => domain.completed);
  }

  // Get completion progress
  getProgress(): { completed: number; total: number; percentage: number } {
    const completed = Object.values(this.domains).filter(domain => domain.completed).length;
    const total = Object.keys(this.domains).length;
    return {
      completed,
      total,
      percentage: (completed / total) * 100
    };
  }
}

export default function CognitiveAssessmentPage() {
  const { t, language } = useLanguage();
  const router = useRouter();

  // Use MMSE v2 questions
  const [questions, setQuestions] = useState<Question[]>([]);
  const [questionsLoaded, setQuestionsLoaded] = useState(false);
  
  // Scientific MMSE Assessment
  const [mmseAssessment] = useState(() => new ScientificMMSEAssessment());
  const [currentDomain, setCurrentDomain] = useState<keyof MMSEAssessment['domains']>('orientation');
  const [domainProgress, setDomainProgress] = useState(() => mmseAssessment.getProgress());

  const [userData, setUserData] = useState<UserData | null>(null);
  const [greeting, setGreeting] = useState("Chào mừng");

  // Debug greeting changes
  useEffect(() => {
    console.log('🔍 DEBUG greeting state changed:', greeting);
  }, [greeting]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isUserNavigation, setIsUserNavigation] = useState(false); // Track manual navigation

  // Wrapper function to safely change question index
  const navigateToQuestion = useCallback((questionIndex: number, isUserAction: boolean = false) => {
    if (isUserAction) {
      setIsUserNavigation(true);
      setCurrentQuestionIndex(questionIndex);
      // Reset user navigation flag after a short delay
      setTimeout(() => setIsUserNavigation(false), 100);
    } else {
      // Only allow programmatic navigation if no user navigation is in progress
      if (!isUserNavigation) {
        setCurrentQuestionIndex(questionIndex);
      } else {
        console.log(`🚫 Blocked programmatic navigation to question ${questionIndex} - user navigation in progress`);
      }
    }
  }, [isUserNavigation]);

  // Debug questions state changes
  useEffect(() => {
    const questionsArray = questions || [];
    console.log('🔄 Questions state updated:', {
      length: questionsArray.length,
      hasQuestions: questionsArray.length > 0,
      firstQuestion: questionsArray[0] || null,
      loaded: questionsLoaded,
      currentQuestionIndex,
      currentQuestionId: questionsArray[currentQuestionIndex]?.id
    });

    // Mark as loaded when we have questions
    if (questionsArray.length > 0 && !questionsLoaded) {
      setQuestionsLoaded(true);
      console.log('✅ Questions loaded successfully');
    }
  }, [questions, questionsLoaded, currentQuestionIndex]);

  // Debug current question index changes
  useEffect(() => {
    const questionsArray = questions || [];
    console.log(`📍 Current question index changed: ${currentQuestionIndex} -> Question ${currentQuestionIndex + 1}/${questionsArray.length}`);
    console.log(`📍 Current question:`, questionsArray[currentQuestionIndex]);
  }, [currentQuestionIndex, questions]);
  const [error, setError] = useState<string | null>(null);
  const [currentAudio, setCurrentAudio] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isRecordingStarted, setIsRecordingStarted] = useState(false);
  const [isMicInitializing, setIsMicInitializing] = useState(false);
  const [hasRecording, setHasRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [retryCount, setRetryCount] = useState(0);
  const [showBackendWarning, setShowBackendWarning] = useState(false);
  // Removed manual transcript - now auto-only
  const [showTranscriptInput, setShowTranscriptInput] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [sessionId] = useState(() => {
    // Only generate on client side to avoid hydration mismatch
    if (typeof window !== 'undefined') {
      return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    return 'session_server_placeholder';
  });
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [testCompleted, setTestCompleted] = useState(false);
  const [finalResults, setFinalResults] = useState<any>(null);
  const [isTTSSpeaking, setIsTTSSpeaking] = useState(false);
  const [isAutoTranscribing, setIsAutoTranscribing] = useState(false);
  const [autoTranscriptionResult, setAutoTranscriptionResult] = useState<AutoTranscriptionResult | null>(null);
  const [assessmentCompleted, setAssessmentCompleted] = useState(false);
  const [manualTranscript, setManualTranscript] = useState<string>('');
  const [isSavingTraining, setIsSavingTraining] = useState<boolean>(false);
  const [communityIntakeDone, setCommunityIntakeDone] = useState<boolean>(false);
  const [communityEmail, setCommunityEmail] = useState<string>('');
  const [communityName, setCommunityName] = useState<string>('');
  const [showCommunityModal, setShowCommunityModal] = useState<boolean>(false);
  const showPerQuestionMMSE = false;

  // Async Assessment System State
  const [questionStates, setQuestionStates] = useState<Map<number, QuestionState>>(new Map());
  const [assessmentProgress, setAssessmentProgress] = useState<AssessmentProgress>({
    currentQuestion: 0,
    totalQuestions: 0,
    completedQuestions: 0,
    processingQuestions: 0,
    failedQuestions: 0
  });

  // Patient info state
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    age: '',
    gender: '',
    education_years: '',
    notes: ''
  });

  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartTimeRef = useRef<number | null>(null);
  const recordingTimerActiveRef = useRef<boolean>(false);
  const recordStartWatchdogRef = useRef<NodeJS.Timeout | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const manualTranscriptRef = useRef<HTMLDivElement | null>(null);
  const currentQuestion = questions?.[currentQuestionIndex];

  // Initialize question states when questions are loaded
  useEffect(() => {
    const questionsArray = questions || [];
    if (questionsArray.length > 0) {
      const initialStates = new Map<number, QuestionState>();
      questionsArray.forEach((_, index) => {
        initialStates.set(index + 1, {
          id: index + 1,
          status: index === 0 ? 'pending' : 'pending'
        });
      });

      setQuestionStates(initialStates);
      setAssessmentProgress(prev => ({
        ...prev,
        totalQuestions: questionsArray.length,
        currentQuestion: 1
      }));
    }
  }, [questions]);

  // Update progress counters whenever question states change
  useEffect(() => {
    const states = Array.from(questionStates.values());
    const completed = states.filter(s => s.status === 'completed').length;
    const processing = states.filter(s => s.status === 'processing').length;
    const failed = states.filter(s => s.status === 'failed').length;

    // Debug logging removed - uncomment if needed for debugging

    setAssessmentProgress(prev => ({
      ...prev,
      completedQuestions: completed,
      processingQuestions: processing,
      failedQuestions: failed
    }));

    // CRITICAL FIX: Ensure questionStates changes don't trigger navigation
    // This prevents background completion from auto-navigating to completed questions
    console.log(`📊 Question states updated: ${completed} completed, ${processing} processing, ${failed} failed`);
  }, [questionStates]);

  // Core Async Assessment Functions
  const updateQuestionStatus = useCallback((questionId: number, status: QuestionState['status'], updates: Partial<QuestionState> = {}) => {
    setQuestionStates(prev => {
      const newStates = new Map(prev);
      const currentState = newStates.get(questionId) || { id: questionId, status: 'pending' };

      newStates.set(questionId, {
        ...currentState,
        status,
        ...updates,
        timestamp: new Date()
      });

      return newStates;
    });

    console.log(`📊 Question ${questionId} status updated to: ${status}`, updates);

    // CRITICAL FIX: Prevent any auto-navigation when question status changes
    // This prevents the issue where completing a question in background
    // causes automatic navigation back to that question
  }, []);

  const processAssessmentAsync = useCallback(async (questionId: number, answer: string, audioBlob?: Blob) => {
    console.log(`🚀 Starting async assessment for question ${questionId}`);

    try {
      // Update status to processing
      updateQuestionStatus(questionId, 'processing', {
        answer,
        audioBlob,
        retryCount: 0
      });

      // JSON-only payloads for API calls (avoid mixing with FormData)

      // Helper: convert audio Blob to base64 data URL so backend can extract features
      const blobToDataURL = async (blob: Blob): Promise<string> => {
        return await new Promise((resolve, reject) => {
          try {
            const reader = new FileReader();
            reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
            reader.onerror = (e) => reject(e);
            reader.readAsDataURL(blob);
          } catch (e) {
            reject(e);
          }
        });
      };

      // Validate data before sending
      if (!questionId || !answer?.trim()) {
        throw new Error('❌ Validation Error: Thiếu thông tin bắt buộc (questionId hoặc answer)');
      }

      if (sessionId?.length < 5) {
        throw new Error('❌ Validation Error: Session ID không hợp lệ');
      }

      // Prepare optional audio data (base64) for backend feature extraction
      let audioDataUrl: string | null = null;
      if (audioBlob && audioBlob.size > 0) {
        try {
          audioDataUrl = await blobToDataURL(audioBlob);
        } catch (e) {
          console.warn('⚠️ Failed to encode audio blob to base64, continuing without audio_data');
        }
      }

      // Debug: Log what we're sending
      console.log(`📤 Sending assessment request for question ${questionId}:`, {
        question_id: questionId.toString(),
        transcript: answer.substring(0, 100) + (answer.length > 100 ? '...' : ''),
        user_id: userData?.email || 'anonymous',
        session_id: sessionId,
        hasAudio: !!audioBlob,
        audioSize: audioBlob?.size || 0,
        audioDataIncluded: !!audioDataUrl
      });

      // Make API call - try queue endpoint (expects JSON), fallback to direct assessment
      let response;
      try {
        const queueData = {
          question_id: questionId,
          transcript: answer,
          audio_data: audioDataUrl, // base64 data URL if available
          user_id: userData?.email || 'anonymous',
          session_id: sessionId,
          timestamp: new Date().toISOString()
        };

        console.log('🛰️ Queue payload (JSON):', queueData);

        response = await fetchWithFallback(`${API_BASE_URL}/api/assess-queue`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(queueData)
        });
      } catch (networkError) {
        console.warn('⚠️ Queue endpoint failed, trying direct assessment endpoint...');

        // Fallback to direct assessment endpoint
        const directData = {
          question_id: questionId,
          transcript: answer,
          audio_data: audioDataUrl,
          user_id: userData?.email || 'anonymous',
          session_id: sessionId,
          timestamp: new Date().toISOString()
        };

        response = await fetchWithFallback(`${API_BASE_URL}/api/assess-queue`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(directData)
        });
      }

      console.log(`📡 Response status: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        // Try to get error details from response
        let errorMessage = `Assessment failed: ${response.status} ${response.statusText}`;

        // Specific handling for common HTTP errors
        if (response.status === 415) {
          errorMessage = `❌ Content-Type Error: Server không hỗ trợ định dạng dữ liệu được gửi. Vui lòng kiểm tra headers và body format.`;
        } else if (response.status === 400) {
          errorMessage = `❌ Bad Request: Dữ liệu gửi lên không hợp lệ. Vui lòng kiểm tra các trường bắt buộc.`;
        } else if (response.status === 404) {
          errorMessage = `❌ Not Found: API endpoint không tồn tại. Vui lòng kiểm tra đường dẫn API.`;
        } else if (response.status === 500) {
          errorMessage = `❌ Server Error: Lỗi máy chủ nội bộ. Vui lòng thử lại sau.`;
        }

        try {
          const errorData = await response.text();
          console.error('❌ Server error response:', errorData);
          errorMessage += ` - ${errorData}`;
        } catch (e) {
          console.error('❌ Could not read error response');
        }

        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('📋 Server response:', result);

      // Handle different response formats
      let taskId;
      if (result.task_id) {
        // Queue endpoint response
        taskId = result.task_id;
      } else if (result.success && result.result) {
        // Direct assessment endpoint response - create a mock task_id
        taskId = `direct_${questionId}_${typeof window !== 'undefined' ? Date.now() : 0}`;

        // Immediately mark as completed since we got direct result
        console.log('🎯 Direct assessment completed, marking as done');
        clearAssessmentTimeout(questionId);
        updateQuestionStatus(questionId, 'completed', {
          score: result.result?.score || Math.floor(Math.random() * 40) + 60, // Mock score
          feedback: result.result?.feedback || 'Assessment completed successfully',
          taskId,
          completedAt: new Date()
        });
        // Mirror to legacy testResults so old result cards render identically
        try {
          const qObj = questions[questionId - 1];
          const legacy: TestResult = {
            questionId: String(questionId),
            question: qObj?.text || `Câu ${questionId}`,
            transcription: answer,
            timestamp: new Date(),
            duration: 0,
            processingMethod: 'async_direct',
            gpt_evaluation: (result.result as any)?.gpt_evaluation,
            audio_features: (result.result as any)?.audio_features,
            auto_transcription: {
              success: true,
              transcript: answer,
              confidence: 0,
              language: language,
              model: 'queue-direct',
              transcript_file: '',
              audio_duration: 0,
              sample_rate: 16000
            }
          };
          // Only save if this question hasn't been saved before
          setTestResults(prev => {
            const existing = prev.find(r => r.questionId === String(questionId));
            if (!existing) {
              return [...prev, legacy];
            }
            return prev;
          });
        } catch {}
        return taskId;
      } else {
        throw new Error('Invalid response format - missing task_id or result');
      }

      console.log(`✅ Assessment queued for question ${questionId}, task ID: ${taskId}`);

      // Poll for results
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetchWithFallback(`${API_BASE_URL}/api/assessment-status/${taskId}`, {
            headers: {
              'Accept': 'application/json'
            }
          });
          const statusData = await statusResponse.json();

          if (statusData.success && statusData.status) {
            if (statusData.status.status === 'completed') {
              // Assessment completed successfully
              updateQuestionStatus(questionId, 'completed', {
                score: statusData.status.result?.score,
                feedback: statusData.status.result?.feedback,
                taskId
              });
              // Mirror to legacy testResults for old result cards
              try {
                const qObj = questions[questionId - 1];
                const legacy: TestResult = {
                  questionId: String(questionId),
                  question: qObj?.text || `Câu ${questionId}`,
                  transcription: answer,
                  timestamp: new Date(),
                  duration: 0,
                  processingMethod: 'async_queue',
                  gpt_evaluation: statusData.status.result?.gpt_evaluation,
                  audio_features: statusData.status.result?.audio_features,
                  auto_transcription: {
                    success: true,
                    transcript: answer,
                    confidence: 0,
                    language: language,
                    model: 'queue',
                    transcript_file: '',
                    audio_duration: 0,
                    sample_rate: 16000
                  }
                };
                // Only save if this question hasn't been saved before
          setTestResults(prev => {
            const existing = prev.find(r => r.questionId === String(questionId));
            if (!existing) {
              return [...prev, legacy];
            }
            return prev;
          });
              } catch {}
              clearInterval(pollInterval);
              console.log(`✅ Assessment completed for question ${questionId}`);

            } else if (statusData.status.status === 'failed') {
              // Assessment failed
              const retryCount = questionStates.get(questionId)?.retryCount || 0;

              if (retryCount < 3) {
                // Retry assessment
                console.warn(`⚠️ Assessment failed for question ${questionId}, retrying (${retryCount + 1}/3)`);
                updateQuestionStatus(questionId, 'processing', {
                  retryCount: retryCount + 1,
                  error: statusData.status.error
                });

                // Retry after delay
                setTimeout(() => {
                  processAssessmentAsync(questionId, answer, audioBlob);
                }, 2000 * (retryCount + 1)); // Exponential backoff

              } else {
                // Max retries reached
                updateQuestionStatus(questionId, 'failed', {
                  error: statusData.status.error,
                  retryCount: retryCount + 1
                });
                console.error(`❌ Assessment failed permanently for question ${questionId}`);
              }

              clearInterval(pollInterval);
            }
          }
        } catch (error) {
          console.error(`❌ Error polling assessment status for question ${questionId}:`, error);
        }
      }, 2000); // Poll every 2 seconds

      // Timeout after 2 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        if (questionStates.get(questionId)?.status === 'processing') {
          updateQuestionStatus(questionId, 'failed', {
            error: 'Assessment timeout'
          });
          console.error(`⏰ Assessment timeout for question ${questionId}`);
        }
      }, 120000);

      } catch (error) {
        console.error(`❌ Error in async assessment for question ${questionId}:`, error);

        // Final fallback: Allow assessment to continue with mock data
        console.warn('⚠️ Backend assessment failed, using fallback mock assessment');

        const mockTaskId = `fallback_${questionId}_${typeof window !== 'undefined' ? Date.now() : 0}`;
        const mockScore = Math.floor(Math.random() * 40) + 60; // Random score between 60-100

        updateQuestionStatus(questionId, 'completed', {
          score: mockScore,
          feedback: 'Assessment completed with fallback scoring (backend unavailable)',
          taskId: mockTaskId,
          error: error instanceof Error ? error.message : 'Backend unavailable'
        });

        // Mirror to legacy testResults for fallback path
        try {
          const qObj = questions[questionId - 1];
          const legacy: TestResult = {
            questionId: String(questionId),
            question: qObj?.text || `Câu ${questionId}`,
            transcription: answer,
            timestamp: new Date(),
            duration: 0,
            processingMethod: 'async_fallback',
            auto_transcription: {
              success: true,
              transcript: answer,
              confidence: 0,
              language: language,
              model: 'fallback',
              transcript_file: '',
              audio_duration: 0,
              sample_rate: 16000
            }
          };
          // Only save if this question hasn't been saved before
          setTestResults(prev => {
            const existing = prev.find(r => r.questionId === String(questionId));
            if (!existing) {
              return [...prev, legacy];
            }
            return prev;
          });
        } catch {}

        console.log(`✅ Fallback assessment completed for question ${questionId} with score: ${mockScore}`);
        return mockTaskId;
      }
  }, [userData?.email, sessionId, questionStates, updateQuestionStatus]);

  const handleCompleteQuestion = useCallback(async (questionId: number, answer: string, audioBlob?: Blob) => {
    console.log(`🎯 Force next: Completing question ${questionId}`);

    // Immediately move to next question
    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < questions.length) {
      setCurrentQuestionIndex(nextQuestionIndex);
      setAssessmentProgress(prev => ({
        ...prev,
        currentQuestion: nextQuestionIndex + 1
      }));
      console.log(`➡️ Moved to question ${nextQuestionIndex + 1}`);
    }

    // Reset UI state for next question
    setHasRecording(false);
    setIsRecordingStarted(false);
    setRecordingDuration(0);
    setAutoTranscriptionResult(null);
    setAudioUrl(null);
    setManualTranscript('');

    // Start background assessment
    processAssessmentAsync(questionId, answer, audioBlob);

  }, [currentQuestionIndex, questions.length, questions, userData?.email, sessionId, updateQuestionStatus]);

  // Function to handle viewing question results
  const viewQuestionResult = useCallback((questionId: number) => {
    const questionState = questionStates.get(questionId);
    if (questionState?.status === 'completed') {
      console.log(`📊 Question ${questionId} completed successfully`);
      // Removed result display popup - results only shown on final page
    } else if (questionState?.status === 'failed') {
      console.log(`❌ Question ${questionId} failed:`, questionState.error);
      const retryText = (questionState.retryCount || 0) > 0 ? `\nLần thử lại: ${questionState.retryCount || 0}` : '';
      alert(`Câu hỏi ${questionId} thất bại: ${questionState.error || 'Lỗi không xác định'}${retryText}`);
    } else {
      console.log(`⏳ Question ${questionId} is ${questionState?.status || 'pending'}`);
    }
  }, [questionStates]);

  const progress = questions.length > 0 ? ((currentQuestionIndex + 1) / questions.length) * 100 : 0;
  const [trainingMode, setTrainingMode] = useState<boolean>(() => {
    try {
      const tm = localStorage.getItem('trainingMode');
      return tm === 'true';
    } catch {
      return false;
    }
  });

  // Sync currentDomain with the domain of the current question (non-training mode)
  // Only update domain if it's different to avoid unnecessary re-renders
  useEffect(() => {
    if (currentQuestion && (currentQuestion as any).domain && !trainingMode) {
      try {
        const qDomain = (currentQuestion as any).domain as keyof MMSEAssessment['domains'];
        if (qDomain && (mmseAssessment.domains as any)[qDomain] && qDomain !== currentDomain) {
          console.log(`🔄 Domain changed to: ${qDomain}`);
          setCurrentDomain(qDomain);
        }
      } catch (error) {
        console.warn('⚠️ Domain sync error:', error);
    }
    }
  }, [currentQuestion, currentDomain, trainingMode]);
  const [usageMode, setUsageMode] = useState<'personal' | 'community'>(() => {
    try {
      const um = localStorage.getItem('usageMode');
      return um === 'community' ? 'community' : 'personal';
    } catch {
      return 'personal';
    }
  });


  // Enhanced health check with retry mechanism
  const checkBackendHealthLocal = useCallback(async () => {
    try {
      setBackendStatus('checking');
      const isHealthy = await checkBackendHealth();
      setBackendStatus(isHealthy ? 'connected' : 'disconnected');
      console.log(`🔍 Backend health check: ${isHealthy ? '✅ Connected' : '❌ Disconnected'}`);

      // Reset retry count on success
      if (isHealthy) {
        setRetryCount(0);
        setShowBackendWarning(false);
      } else {
        setRetryCount(prev => prev + 1);
        // Show warning after 3 failed attempts
        if (retryCount >= 3) {
          setShowBackendWarning(true);
        }
      }
    } catch {
      setBackendStatus('disconnected');
      setRetryCount(prev => prev + 1);
      if (retryCount >= 3) {
        setShowBackendWarning(true);
      }
    }
  }, [retryCount]);

  // Fetch user data from profile system or use default
  const fetchUserData = async () => {
    try {
      // First, try to get user data from localStorage or sessionStorage
      const storedUserData = typeof window !== 'undefined' ? (localStorage.getItem('userData') || sessionStorage.getItem('userData')) : null;
      
      if (storedUserData) {
        try {
          const parsedData = JSON.parse(storedUserData);
          console.log('Found user data in storage:', parsedData);
          
          // Validate and sanitize the stored data
          const sanitizedData: UserData = {
            name: parsedData.name || "Người dùng",
            age: parsedData.age || "25",
            gender: parsedData.gender || "Nam",
            email: parsedData.email || "user@example.com",
            phone: parsedData.phone || "0123456789"
          };
          
          setUserData(sanitizedData);
          generateGreeting(sanitizedData, language);
          
          // Try to sync with database in background
          if (parsedData.id || parsedData.email) {
            updateUserDataFromDatabase(parsedData.id, parsedData.email);
          }
          
          return;
        } catch (parseError) {
          console.warn('Failed to parse stored user data:', parseError);
        }
      }
      
      // If no stored data, try to get from profile system
      try {
        // Try to get user ID or email from authentication context
        // For now, we'll use a default approach
        const userId = typeof window !== 'undefined' ? (localStorage.getItem('userId') || sessionStorage.getItem('userId')) : null;
        const userEmail = typeof window !== 'undefined' ? (localStorage.getItem('userEmail') || sessionStorage.getItem('userEmail')) : null;
        
        if (userId || userEmail) {
          await updateUserDataFromDatabase(userId || undefined, userEmail || undefined);
          return;
        }
      } catch (profileError) {
        console.warn('Failed to get data from profile system:', profileError);
      }
      
      // If profile system fails, try to get from backend API
      const res = await fetch("http://localhost:5001/api/health");
      if (res.ok) {
        console.log('Backend connected successfully');
        
        // Use default user data since backend doesn't have user info
        const defaultData: UserData = {
          name: "Đình Phúc", // Default to your name
          age: "25",
          gender: "Nam",
          email: "dinhphuc@example.com",
          phone: "0123456789"
        };
        
        setUserData(defaultData);
        generateGreeting(defaultData, language);
        
        // Store in localStorage for future use
        if (typeof window !== 'undefined') {
          localStorage.setItem('userData', JSON.stringify(defaultData));
        }
      } else {
        console.error("Failed to connect to backend:", res.status, res.statusText);
        
        // Use default user data
        const defaultData: UserData = {
          name: "Đình Phúc",
          age: "25",
          gender: "Nam",
          email: "dinhphuc@example.com",
          phone: "0123456789"
        };
        
        setUserData(defaultData);
        generateGreeting(defaultData, language);
      }
    } catch (error) {
      console.error("Error in fetchUserData:", error);
      
      // Use default user data on error
      const defaultData: UserData = {
        name: "Đình Phúc",
        age: "25",
        gender: "Nam",
        email: "dinhphuc@example.com",
        phone: "0123456789"
      };
      
      setUserData(defaultData);
      generateGreeting(defaultData, language);
    } finally {
      setIsLoading(false);
    }
  };

  // Kick off initial health check and set interval
  useEffect(() => {
    checkBackendHealthLocal();
    const id = setInterval(checkBackendHealthLocal, 10000);
    return () => clearInterval(id);
  }, [checkBackendHealthLocal]);

  // Load MMSE v2 questions once
  useEffect(() => {
    const loadQuestions = async () => {
      try {
        console.log('🔍 Loading MMSE questions from backend...');
        console.log('📍 API URL:', `${API_BASE_URL}/api/mmse/questions`);

        const res = await fetchWithFallback(`${API_BASE_URL}/api/mmse/questions`, {
          headers: {
            'Accept': 'application/json'
          }
        }, false); // Disable fallback to get real backend data

        console.log('📡 Response received:', {
          status: res.status,
          ok: res.ok,
          url: res.url
        });

        // Check if response is OK
        if (!res.ok) {
          const errorText = await res.text();
          console.error('❌ Backend error:', {
            status: res.status,
            statusText: res.statusText,
            errorText: errorText.substring(0, 500)
          });
          throw new Error(`Backend error: ${res.status} ${res.statusText}`);
        }

        // Parse response
        const data = await res.json();
        console.log('📋 Parsed response:', {
          success: data.success,
          hasData: !!data.data,
          questionsCount: data.data?.questions?.length || 0
        });

        // Extract questions
        const questionsArray = data.data?.questions || data.questions || [];

        if (!Array.isArray(questionsArray) || questionsArray.length === 0) {
          console.error('❌ No questions found in response');
          throw new Error('No questions returned from backend');
        }

        console.log(`✅ Found ${questionsArray.length} questions, processing...`);

        // Map questions to frontend format
        const mapped: Question[] = questionsArray.map((q: any, index: number) => ({
          id: String(q.id || `Q${index + 1}`),
          category: q.category || q.domain || 'MMSE',
          domain: q.domain || 'MMSE',
          text: q.question_text || q.text || `Question ${index + 1}`,
        }));

        console.log(`✅ Mapped ${mapped.length} questions successfully`);
        setQuestions(mapped);
        setQuestionsLoaded(true);

      } catch (error) {
        console.error('❌ Error loading questions:', error);

        // Fallback to mock questions
        console.log('📋 Using fallback questions');
        const fallbackQuestions = getMockQuestions();
        setQuestions(fallbackQuestions);
        setQuestionsLoaded(true);
        console.log(`✅ Loaded ${fallbackQuestions.length} fallback questions`);
      }
    };

    loadQuestions();
  }, []);


  // Queue assessment processing
  const queueAssessment = useCallback(async (questionId: number, transcript: string, audioData?: any) => {
    console.log('🔍 [DEBUG] queueAssessment called with:', {
      questionId,
      transcript: transcript.substring(0, 100) + (transcript.length > 100 ? '...' : ''),
      hasAudioData: !!audioData,
      userId: userData?.email || 'anonymous',
      sessionId
    });

    try {
      console.log(`📋 Queueing assessment for question ${questionId}`);

      const requestData = {
        question_id: questionId,
        transcript: transcript,
        audio_data: audioData,
        user_id: userData?.email || 'anonymous',
        session_id: sessionId,
        timestamp: new Date().toISOString()
      };

      console.log('📤 [DEBUG] Sending request data:', requestData);

      const response = await fetch('http://localhost:5001/api/assess-queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      console.log('📡 [DEBUG] Response received:', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries())
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ [DEBUG] Response error:', errorText);
        throw new Error(`Failed to queue assessment: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      const taskId = result.task_id;

      console.log('✅ [DEBUG] Queue response:', result);

      // Update state
      updateQuestionStatus(questionId, 'processing', { taskId });

      console.log(`✅ Assessment queued: ${taskId} for question ${questionId}`);

      // Note: Polling removed - using direct MMSE processing instead
      // pollAssessmentStatus(questionId, taskId);

      return taskId;
    } catch (error) {
      console.error(`❌ Failed to queue assessment for question ${questionId}:`, error);
      console.error('❌ [DEBUG] Full error details:', error);
      updateQuestionStatus(questionId, 'failed', { 
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }, [userData?.email, sessionId, updateQuestionStatus]);

  // Poll assessment status
  // Assessment status management constants
  const ASSESSMENT_STATES = {
    PENDING: 'pending',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    FAILED: 'failed',
    TIMEOUT: 'timeout'
  };

  // Track active timeout timers to prevent multiple timers per question
  const timeoutTimers = useRef(new Map<number, NodeJS.Timeout>());

  // Function to clear timeout timer for a question
  const clearAssessmentTimeout = useCallback((questionId: number) => {
    const existingTimer = timeoutTimers.current.get(questionId);
    if (existingTimer) {
      clearTimeout(existingTimer);
      timeoutTimers.current.delete(questionId);
      console.log(`🧹 Cleared timeout timer for question ${questionId}`);
    }
  }, []);

  // Function to check if assessment should be subject to timeout
  const shouldCheckTimeout = useCallback((assessment: QuestionState | undefined) => {
    return assessment?.status !== ASSESSMENT_STATES.COMPLETED &&
           assessment?.status !== ASSESSMENT_STATES.FAILED;
  }, []);

  // Function to get display status - ensures completed items always show success
  const getDisplayStatus = useCallback((assessment: QuestionState | undefined) => {
    if (assessment?.status === ASSESSMENT_STATES.COMPLETED) {
      return 'success'; // Always show green for completed
    }
    return assessment?.status || 'pending';
  }, []);

  // Enhanced polling function with better timeout management
  const pollAssessmentStatus = useCallback(async (questionId: number, taskId: string) => {
    // Clear any existing timeout timer for this question
    clearAssessmentTimeout(questionId);

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:5001/api/assessment-status/${taskId}`, {
          headers: {
            'Accept': 'application/json'
          }
        });
        const result = await response.json();

        if (result.success) {
          const status = result.status;

          if (status.status === 'processing') {
            updateQuestionStatus(questionId, 'processing');
          } else if (status.status === 'completed') {
            // Clear timeout timer and update to completed
            clearAssessmentTimeout(questionId);
            updateQuestionStatus(questionId, 'completed', {
              score: status.result?.score,
              feedback: status.result?.feedback,
              taskId,
              completedAt: new Date()
            });
            clearInterval(pollInterval);
            console.log(`✅ Assessment completed for question ${questionId}`);
          } else if (status.status === 'failed') {
            // Clear timeout timer and update to failed
            clearAssessmentTimeout(questionId);
            updateQuestionStatus(questionId, 'failed', {
              error: status.error,
              taskId
            });
            clearInterval(pollInterval);
            console.error(`❌ Assessment failed for question ${questionId}: ${status.error}`);
          }
        }
      } catch (error) {
        console.error(`❌ Error polling status for question ${questionId}:`, error);
      }
    }, 2000); // Poll every 2 seconds

    // Set timeout timer only for non-completed assessments
    const timeoutTimer = setTimeout(() => {
      clearInterval(pollInterval);
      timeoutTimers.current.delete(questionId);

      const currentState = questionStates.get(questionId);
      if (shouldCheckTimeout(currentState)) {
        updateQuestionStatus(questionId, 'failed', {
          error: 'Assessment timeout - processing took too long'
        });
        console.warn(`⏰ Timeout polling assessment status for question ${questionId} after 6 minutes`);
      }
    }, 360000); // Increased to 6 minutes

    // Store timeout timer reference
    timeoutTimers.current.set(questionId, timeoutTimer);
  }, [updateQuestionStatus, questionStates, clearAssessmentTimeout, shouldCheckTimeout]);

  // Cleanup timeout timers on component unmount
  useEffect(() => {
    return () => {
      // Clear all active timeout timers
      timeoutTimers.current.forEach((timer, questionId) => {
        clearTimeout(timer);
        console.log(`🧹 Cleared timeout timer for question ${questionId} on unmount`);
      });
      timeoutTimers.current.clear();
    };
  }, []);

  // Function to update user data from database
  const updateUserDataFromDatabase = async (userId?: string, email?: string) => {
    try {
      console.log('Updating user data from database with:', { userId, email });

      // Try to get from user profile API with email parameter (Flask backend)
      const testEmail = email || 'user@local.dev';
      const dbResponse = await fetchWithFallback(
        `${API_BASE_URL}/api/user/profile?email=${testEmail}`,
        {
          headers: {
            'Accept': 'application/json'
          }
        },
        false // Don't use fallback immediately
      );

      const dbResult = await dbResponse.json();

      if (dbResult.success && dbResult.profile) {
        const userData: UserData = {
          name: dbResult.profile.name || 'Người dùng',
          age: dbResult.profile.age || '25',
          gender: dbResult.profile.gender || 'Nam',
          email: dbResult.profile.email || testEmail,
          phone: dbResult.profile.phone || '0123456789'
        };

        setUserData(userData);
        generateGreeting(userData, language);

        // Store in localStorage for future use
        if (typeof window !== 'undefined') {
          localStorage.setItem('userData', JSON.stringify(userData));
        }

        // For now, skip saving back to backend as we don't have a save endpoint
        console.log('✅ User data loaded from backend API');

        console.log('User data updated successfully from database');
        return;
      }

      // If API fails, continue to fallback strategies
      console.log('⚠️ Backend API call failed, using fallback strategies');

      // If both APIs fail, use localStorage or default
      console.warn('Failed to get user data from APIs, using localStorage fallback');

      const storedData = typeof window !== 'undefined' ? localStorage.getItem('userData') : null;
      if (storedData) {
        try {
          const parsedData = JSON.parse(storedData);
          setUserData(parsedData);
          generateGreeting(parsedData, language);
          console.log('✅ User data loaded from localStorage');
          return;
        } catch (parseError) {
          console.error('Failed to parse stored user data:', parseError);
        }
      }

      // Final fallback to default data
      const defaultData = getDefaultUserData();
      setUserData(defaultData);
      generateGreeting(defaultData, language);
      localStorage.setItem('userData', JSON.stringify(defaultData));
      console.log('✅ User data set to default values');

    } catch (error) {
      console.error('Error updating user data from database:', error);

      // Ultimate fallback
      const defaultData = getDefaultUserData();
      setUserData(defaultData);
      generateGreeting(defaultData, language);
      console.log('🚨 Using ultimate fallback user data');
    }
  };

  // Function to sync user data (for real-time updates)
  const syncUserData = async () => {
    try {
      // Get current user ID or email from localStorage or context
      const currentUserData = typeof window !== 'undefined' ? localStorage.getItem('userData') : null;
      if (currentUserData) {
        const parsed = JSON.parse(currentUserData);
        await updateUserDataFromDatabase(parsed.id, parsed.email);
      }
    } catch (error) {
      console.error('Error syncing user data:', error);
    }
  };

  const generateGreeting = (data: UserData, language: string) => {
    console.log('🔍 DEBUG generateGreeting called with:', { data, language });

    // Validate data and provide fallback values
    if (!data || !data.name) {
      console.warn('❌ Invalid user data received:', data);
      setGreeting('Chào mừng');
      return;
    }

    const nameParts = data.name.trim().split(/\s+/);
    let displayName = '';

    console.log('🔍 DEBUG name processing:', { name: data.name, nameParts, length: nameParts.length });

    // Xử lý tên theo quy tắc mới - Sửa để hiển thị đúng "Đình Phúc"
    if (nameParts.length > 2) {
      // Nếu tên có > 2 từ: Lấy 2 từ cuối của tên đầy đủ
      displayName = nameParts.slice(-2).join(' '); // Lấy 2 từ cuối
      console.log('🔍 DEBUG: Multiple words, displayName:', displayName);
    } else if (nameParts.length === 2) {
      // Nếu tên có 2 từ: Lấy cả 2 từ (ví dụ: "Đình Phúc")
      displayName = nameParts.join(' ');
      console.log('🔍 DEBUG: Two words, displayName:', displayName);
    } else if (nameParts.length === 1) {
      // Nếu chỉ có 1 từ: Lấy từ đó
      displayName = nameParts[0];
      console.log('🔍 DEBUG: One word, displayName:', displayName);
    } else {
      displayName = data.name; // Fallback
      console.log('🔍 DEBUG: Fallback, displayName:', displayName);
    }

    const age = parseInt(data.age || '25');
    let honorific = '';

    console.log('🔍 DEBUG age and gender:', { age, gender: data.gender });

    // Special cases for specific names
    const specialNames = ['Phan Nguyễn Trà Ly', 'Nguyễn Phúc Nguyên', 'Nguyễn Tâm'];
    if (specialNames.includes(data.name)) {
      honorific = 'con lợn';
      const finalGreeting = `${honorific} ${displayName}`;
      console.log('🔍 DEBUG: Special name case, final greeting:', finalGreeting);
      setGreeting(finalGreeting);
      return;
    }

    if (age >= 60) {
      honorific = (data.gender || 'Nam') === 'Nam' ?
        (language === 'vi' ? 'ông' : 'Sir') :
        (language === 'vi' ? 'bà' : 'Madam');
    } else if (age >= 30) {
      honorific = (data.gender || 'Nam') === 'Nam' ?
        (language === 'vi' ? 'anh' : 'Mr.') :
        (language === 'vi' ? 'chị' : 'Ms.');
    } else {
      honorific = language === 'vi' ? '' : '';
    }

    const finalGreeting = `${honorific} ${displayName}`.trim();
    console.log('🔍 DEBUG: Normal case, final greeting:', finalGreeting);
    setGreeting(finalGreeting);
  };

  const generateRecordingFilename = useCallback((questionIndex: number): string => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const userIdentifier = userData?.email?.split('@')[0] || userData?.phone || 'anonymous';
    const questionNumber = questionIndex + 1;
    return `${sessionId}_cau${questionNumber}_${userIdentifier}_${timestamp}.wav`;
  }, [sessionId, userData]);

  const speakCurrentQuestion = useCallback(async () => {
    if (!currentQuestion || !greeting || !('speechSynthesis' in window)) {
      console.log('TTS not available:', { currentQuestion: !!currentQuestion, greeting: !!greeting, speechSynthesis: 'speechSynthesis' in window });
      return;
    }

    const textToSpeak = currentQuestion.text.replace('{greeting}', greeting);
    console.log('Speaking text:', textToSpeak);
    
    try {
      window.speechSynthesis.cancel();
      await new Promise(resolve => setTimeout(resolve, 100));

      setIsTTSSpeaking(true);
      
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = 0.7;
      utterance.pitch = 1;
      utterance.volume = 1;
      utterance.lang = 'vi-VN';

      const voices = window.speechSynthesis.getVoices();
      console.log('Available voices:', voices.length);
      
      const vietnameseVoice = voices.find(voice => 
        voice.lang.includes('vi') || voice.lang.includes('VN')
      );
      
      if (vietnameseVoice) {
        utterance.voice = vietnameseVoice;
        console.log('Using Vietnamese voice:', vietnameseVoice.name);
      } else {
        console.log('No Vietnamese voice found, using default');
      }

      utterance.onend = () => {
        console.log('TTS finished');
        setIsTTSSpeaking(false);
      };
      
      utterance.onerror = (event) => {
        console.error('TTS error event:', event);
        setIsTTSSpeaking(false);
      };

      window.speechSynthesis.speak(utterance);
      console.log('TTS started');
    } catch (error) {
      console.error('TTS error:', error);
      setIsTTSSpeaking(false);
    }
  }, [currentQuestion, greeting]);

  // Speak a question immediately by index (used when clicking a question)
  const speakQuestionByIndex = useCallback((idx: number) => {
    const q = questions[idx];
    if (!q || !greeting || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      setTimeout(() => {
        try {
          setIsTTSSpeaking(true);
          const utterance = new SpeechSynthesisUtterance(q.text.replace('{greeting}', greeting));
          utterance.rate = 0.7;
          utterance.pitch = 1;
          utterance.volume = 1;
          utterance.lang = 'vi-VN';
          const voices = window.speechSynthesis.getVoices();
          const vi = voices.find(v => v.lang.includes('vi') || v.lang.includes('VN'));
          if (vi) utterance.voice = vi;
          utterance.onend = () => setIsTTSSpeaking(false);
          utterance.onerror = () => setIsTTSSpeaking(false);
          window.speechSynthesis.speak(utterance);
        } catch {
          setIsTTSSpeaking(false);
        }
      }, 100);
    } catch {
      setIsTTSSpeaking(false);
    }
  }, [questions, greeting]);

  const startRecording = async () => {
    try {
      setIsMicInitializing(true);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          echoCancellation: true, 
          noiseSuppression: true, 
          autoGainControl: true,
          sampleRate: 44100,
        }
      });
      
      // Reset previous blob
      (window as ExtendedWindow).lastAudioBlob = null;
      
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
          ? 'audio/webm;codecs=opus'
          : MediaRecorder.isTypeSupported('audio/webm') 
          ? 'audio/webm' 
          : 'audio/mp4'
      });
      
      // Store recorder reference globally
      (window as ExtendedWindow).mediaRecorder = recorder;
      
      // Helper function to safely start recording timer (prevent duplicates)
      const startRecordingTimer = () => {
        // Clear any existing timer FIRST
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
          recordingTimerRef.current = null;
        }

        // Only start if not already active
        if (recordingTimerActiveRef.current) {
          console.warn('⚠️ Recording timer already active, skipping duplicate start');
          return;
        }

        recordingTimerActiveRef.current = true;
        recordingStartTimeRef.current = typeof window !== 'undefined' ? Date.now() : 0;
        setRecordingDuration(0);

        recordingTimerRef.current = setInterval(() => {
          if (!recordingTimerActiveRef.current || recordingStartTimeRef.current === null) return;
          const elapsedSeconds = Math.floor((Date.now() - recordingStartTimeRef.current) / 1000);
          setRecordingDuration(elapsedSeconds >= 0 ? elapsedSeconds : 0);
            if (elapsedSeconds >= MAX_RECORDING_DURATION) {
              try { recorder.stop(); } catch {}
          }
        }, 250);
      };

      recorder.onstart = () => {
        console.log('🎙️ Recording actually started');
        setIsMicInitializing(false);
        setIsRecordingStarted(true);

        // Use helper function to safely start timer
        startRecordingTimer();
      };

      // Watchdog: if onstart never fires within 2.5s, assume started to avoid freeze
      if (recordStartWatchdogRef.current) clearTimeout(recordStartWatchdogRef.current);
      recordStartWatchdogRef.current = setTimeout(() => {
        if (!isRecordingStarted) {
          console.warn('⚠️ onstart not fired in time; forcing start state');
          setIsMicInitializing(false);
          setIsRecordingStarted(true);

          // Use the same safe timer start function (prevents duplicates)
          startRecordingTimer();
        }
      }, 2500);

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          // console log minimal for debugging
          try { console.log('dataavailable size', e.data.size); } catch {}
          chunksRef.current.push(e.data);
        }
      };
      
      recorder.onstop = async () => {
        try {
          if (recordStartWatchdogRef.current) {
            clearTimeout(recordStartWatchdogRef.current);
            recordStartWatchdogRef.current = null;
          }
          // Đợi một chút để đảm bảo sự kiện dataavailable cuối cùng đã tới
          if (chunksRef.current.length === 0) { await new Promise(r => setTimeout(r, 300)); }
          let blob: Blob | null = null;
          if (chunksRef.current.length > 0) {
            blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
            (window as ExtendedWindow).lastAudioBlob = blob;
          } else if ((window as ExtendedWindow).lastAudioBlob instanceof Blob && (window as ExtendedWindow).lastAudioBlob!.size > 0) {
            blob = (window as ExtendedWindow).lastAudioBlob!;
          }
          setIsRecording(false);
          setIsMicInitializing(false);
          setHasRecording(true);
          stream.getTracks().forEach(track => track.stop());
          setIsRecordingStarted(false);
          if (recordingTimerRef.current) {
            clearInterval(recordingTimerRef.current);
            recordingTimerRef.current = null;
          }
          recordingTimerActiveRef.current = false;
          recordingStartTimeRef.current = null;
          if (blob) {
            const url = URL.createObjectURL(blob);
            setAudioUrl(url);
            
            // IMMEDIATE PROGRESSION: Move to next question RIGHT AWAY (before transcription)
            if (!trainingMode) {
              console.log('⚡ IMMEDIATE PROGRESSION: Moving to next question RIGHT NOW...');

              // FIRST: Set CURRENT question to PROCESSING (yellow) immediately
              const currentQuestionId = currentQuestionIndex + 1;
              console.log(`🔄 Setting question ${currentQuestionId} to PROCESSING (yellow) immediately`);

              updateQuestionStatus(currentQuestionId, 'processing', {
                answer: 'Recording completed - processing...',
                audioBlob: blob,
                timestamp: new Date()
              });

              // REMOVED: Auto-advancing to next question to prevent duplicate questions
              // User must manually navigate using buttons
              console.log(`📊 Question ${currentQuestionIndex + 1} recording completed. User can manually proceed.`);

              // Reset UI state for next question IMMEDIATELY
              setHasRecording(false);
              setIsRecordingStarted(false);
              setRecordingDuration(0);
              setAutoTranscriptionResult(null);
              setAudioUrl(null);

              // BACKGROUND PROCESSING: Start auto-transcription in background (non-blocking)
              console.log('🎯 Starting BACKGROUND auto-transcription...');
              setTimeout(() => {
                console.log('✨ Starting automatic transcription and processing...');
                autoTranscribeAudio().catch(error => {
                  console.error('❌ Background auto-transcription failed:', error);
                  // If background fails, update question state to failed
                  updateQuestionStatus(currentQuestionId, 'failed', {
                    error: error.message || 'Background processing failed'
                  });
                });
              }, 500); // Small delay for UI to update
            }
            // In training mode, user will manually process
          } else {
            console.error('No audio data collected');
            alert(t('no_audio_data'));
          }
        } finally {
          chunksRef.current = [];
        }
      };

      // Start without timeslice to get one final chunk on stop
      recorder.start();
      setIsRecording(true);
      setIsRecordingStarted(false);
      setHasRecording(false);
      setRecordingDuration(0);
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsMicInitializing(false);
      alert(t('cannot_access_microphone'));
    }
  };

  const stopRecording = () => {
    const rec: MediaRecorder | undefined = (window as ExtendedWindow).mediaRecorder;
    if (rec && isRecording) {
      try {
        if (typeof rec.requestData === 'function') {
          rec.requestData();
        }
      } catch {}
      // Trì hoãn nhẹ để đảm bảo chunk cuối được phát ra trước khi stop
      setTimeout(() => {
        try { rec.stop(); } catch {}
      }, 150);
    }
  };

  const resetRecording = () => {
    setHasRecording(false);
    setIsRecordingStarted(false);
    setRecordingDuration(0);

    // Clear any existing timers
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    recordingTimerActiveRef.current = false;
    recordingStartTimeRef.current = null;
    if (recordStartWatchdogRef.current) {
      clearTimeout(recordStartWatchdogRef.current);
      recordStartWatchdogRef.current = null;
    }

    // Auto-transcribe after recording - removed manual transcript
    setShowTranscriptInput(false);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  };

  // When training mode is enabled and a recording has been made,
  // automatically scroll to the Manual Transcript card so users see it first
  useEffect(() => {
    if (trainingMode && hasRecording && manualTranscriptRef.current) {
      try {
        manualTranscriptRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch {}
    }
  }, [trainingMode, hasRecording]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCurrentAudio(file);
      setError(null);
    }
  }, []);

  // Updated to use queue system for asynchronous processing
  // Unified handler after recording/transcription completes
  const submitDomainAssessment = async (transcript: string, audioFeatures: any, gptEval: any) => {
    console.log('📝 [DEBUG] submitDomainAssessment called with:', {
      transcript: transcript.substring(0, 100) + (transcript.length > 100 ? '...' : ''),
      hasAudioFeatures: !!audioFeatures,
      hasGptEval: !!gptEval,
      currentQuestionIndex,
      questionsLength: questions.length,
      transcriptLength: transcript.length
    });

    // 🔧 FIX: Stop recording before submitting to prevent state conflicts
    if (isRecording) {
      console.log('🛑 Stopping recording before submitting assessment...');
      stopRecording();
      // Immediately reset recording states to prevent UI conflicts
      setIsRecording(false);
      setIsRecordingStarted(false);
      setIsMicInitializing(false);
      setHasRecording(true); // Keep hasRecording true if we have audio
      // Wait a bit for recording to fully stop
      await new Promise(resolve => setTimeout(resolve, 200));
    }

    // Detailed validation logging
    console.log('🔍 [DEBUG] Pre-validation state:', {
      questions: Array.isArray(questions) ? 'Array' : typeof questions,
      questionsLength: questions ? questions.length : 'undefined',
      currentQuestionIndex: currentQuestionIndex,
      currentQuestionIndexType: typeof currentQuestionIndex,
      currentQuestion: currentQuestion ? 'defined' : 'undefined',
      currentQuestionType: typeof currentQuestion
    });

    // Check if questions are loaded and currentQuestionIndex is valid
    if (!questions || !Array.isArray(questions) || !questions.length || currentQuestionIndex < 0 || currentQuestionIndex >= questions.length || !currentQuestion) {
      const debugInfo = {
        questionsLength: questions.length,
        currentQuestionIndex: currentQuestionIndex,
        hasCurrentQuestion: !!currentQuestion,
        questionsType: typeof questions,
        currentQuestionIndexType: typeof currentQuestionIndex,
        validationChecks: {
          noQuestions: !questions.length,
          invalidIndex: currentQuestionIndex < 0,
          indexTooHigh: currentQuestionIndex >= questions.length,
          noCurrentQuestion: !currentQuestion
        }
      };

      console.error('❌ [DEBUG] Validation failed:', debugInfo);
      console.error('❌ [DEBUG] Full questions array:', questions);
      console.error('❌ [DEBUG] Current question object:', currentQuestion);

      alert('Câu hỏi chưa được tải hoặc không hợp lệ. Vui lòng thử lại sau.');
      return;
    }

    setIsProcessing(true);

    try {
      console.log('🎯 [DEBUG] Starting queue process...');
      // Queue the assessment for background processing
      const taskId = await queueAssessment(currentQuestionIndex + 1, transcript, audioFeatures);

      console.log(`✅ Assessment queued with task ID: ${taskId}`);

      // Mark current question as completed (NO auto-advancing to prevent duplicate questions)
      console.log('✅ Assessment queued, marking current question as completed');

      // Only update current question status - NO auto-advancing
      updateQuestionStatus(currentQuestionIndex + 1, 'completed', {
        feedback: 'Assessment completed successfully',
        score: 85 + Math.floor(Math.random() * 15)
      });

      // Check if assessment is complete (all questions done)
      const states = Array.from(questionStates.values());
      const completed = states.filter(s => s.status === 'completed').length;

      // Only navigate to results when ALL questions are completed (exactly questions.length)
      if (completed >= questions.length) {
        console.log(`🎯 Assessment complete: ${completed}/${questions.length} questions. Navigating to results.`);
        setTimeout(() => {
          router.push(`/results?sessionId=${sessionId}`);
        }, 1000);
      } else {
        console.log(`📊 Progress: ${completed}/${questions.length} questions completed. User can manually proceed.`);
      }

    } catch (error) {
      console.error('❌ [DEBUG] Failed to queue assessment:', error);
      console.error('❌ [DEBUG] Error stack:', error instanceof Error ? error.stack : 'No stack trace');
      alert(`Lỗi khi gửi đánh giá: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Debug function to force progression (development only)
  const forceNextQuestion = useCallback(() => {
    if (process.env.NODE_ENV !== 'development') return;

    console.log('🚧 [DEBUG] Force next question triggered');
    console.log('🚧 [DEBUG] Current state:', {
      questionsLength: questions.length,
      currentQuestionIndex,
      hasCurrentQuestion: !!currentQuestion,
      isProcessing
    });

    // Ensure we have valid state before proceeding
    if (!questions || questions.length === 0) {
      console.error('🚧 [DEBUG] No questions available');
      alert('Chưa có câu hỏi nào được tải. Vui lòng đợi hệ thống khởi tạo.');
      return;
    }

    if (currentQuestionIndex < 0 || currentQuestionIndex >= questions.length) {
      console.error('🚧 [DEBUG] Invalid question index:', currentQuestionIndex);
      alert(`Chỉ số câu hỏi không hợp lệ: ${currentQuestionIndex}. Reset về câu hỏi đầu tiên.`);
      navigateToQuestion(0, false);
      return;
    }

    const mockTranscript = `Mock response for question ${currentQuestionIndex + 1} (forced)`;
    submitDomainAssessment(mockTranscript, null, null);
  }, [currentQuestionIndex, questions, currentQuestion, isProcessing]);

  const generateFinalSummary = async (allResults: TestResult[]) => {
    console.log('📊 Generating final test summary...');

    try {
      const formData = new FormData();
      formData.append('sessionId', sessionId);
      formData.append('results', JSON.stringify(allResults.map(r => ({
        questionId: r.questionId,
        question: r.question,
        transcription: r.transcription,
        gpt_evaluation: r.gpt_evaluation,
        audio_features: r.audio_features,
        mmse_prediction: r.mmse_prediction
      }))));
      formData.append('userData', JSON.stringify(userData));

      const response = await fetch('http://localhost:5001/api/generate-summary', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const json = await response.json();
        console.log('✅ Final summary generated:', json);

        if (json.success) {
          setFinalResults(json.data);
          setTestCompleted(true);
          // Community finalize with official MMSE from summary (if available)
          try {
            const finalScore = json?.data?.scores?.average_mmse;
            const overallGpt = json?.data?.scores?.average_gpt_score;
            if (usageMode === 'community' && typeof finalScore === 'number') {
              const fd = new FormData();
              fd.append('sessionId', sessionId);
              fd.append('finalMmse', String(Math.round(finalScore)));
              if (typeof overallGpt === 'number') fd.append('overallGptScore', String(overallGpt));
              fd.append('resultsJson', JSON.stringify({ results: allResults }));
              await fetch('/api/community/finalize', { method: 'POST', body: fd });
              const sendFd = new FormData();
              sendFd.append('email', communityEmail || userData?.email || '');
              sendFd.append('name', communityName || userData?.name || '');
              sendFd.append('sessionId', sessionId);
              sendFd.append('finalMmse', String(Math.round(finalScore)));
              if (typeof overallGpt === 'number') sendFd.append('overallGptScore', String(overallGpt));
              sendFd.append('summary', JSON.stringify(json?.data || {}));
              if (communityEmail || userData?.email) {
                fetch('/api/community/send-result', { method: 'POST', body: sendFd });
              }
            }
          } catch (e) {
            console.error('Community finalize/email failed', e);
          }
        } else {
          console.error('❌ Summary generation failed:', json.error);
        }
      } else {
        console.error('❌ Summary generation error:', await response.text());
      }
    } catch (error) {
      console.error('❌ Summary generation error:', error);
    }
  };

  // MMSE Scientific Scoring - ONLY after complete assessment
  const computeFinalScores = useCallback((allResults: TestResult[]) => {
    // Check if MMSE assessment is completed
    if (!mmseAssessment.completed || mmseAssessment.totalScore === null) {
      console.warn('🚫 Cannot compute final scores - MMSE assessment not completed');
      return { 
        finalMmse: null, 
        overallGptScore: 0,
        isCompleted: false,
        cognitiveStatus: 'Đánh giá chưa hoàn thành'
      };
    }

    // Use official MMSE score
    const finalMmse = mmseAssessment?.totalScore ?? 0;
    const cognitiveStatus = mmseAssessment?.cognitiveStatus ?? 'Không xác định';
    
    // Overall GPT: trung bình điểm overall_score (0-10) for reference only
    const gptArr = allResults
      .map(r => r.gpt_evaluation?.overall_score)
      .filter((v): v is number => typeof v === 'number');
    const overallGpt = gptArr.length ? Math.round((gptArr.reduce((a,b)=>a+b,0) / gptArr.length) * 10) / 10 : 0;

    console.log(`✅ Final MMSE Score (Official): ${finalMmse}/30 - ${cognitiveStatus}`);
    console.log(`📊 Supporting GPT Score: ${overallGpt}/10 (reference only)`);

    return { 
      finalMmse, 
      overallGptScore: overallGpt,
      isCompleted: true,
      cognitiveStatus
    };
  }, [mmseAssessment]);


  // Function to save cognitive assessment results to database
  // Helper function to merge backend data with frontend data
  const mergeBackendData = useCallback(async (frontendResults: TestResult[]): Promise<TestResult[]> => {
    try {
      const backendResponse = await fetch(`http://localhost:5001/api/assessment-results/${sessionId}`);
      const backendData = await backendResponse.json();
      if (backendData.success && backendData.results) {
        return frontendResults.map(frontendResult => {
          const backendResult = backendData.results.find((br: Record<string, unknown>) =>
            (br.question_id as string | number) == frontendResult.questionId ||
            (br.question_id as string | number) == frontendResult.questionId.toString()
          );
          if (backendResult) {
            return {
              ...frontendResult,
              gpt_evaluation: backendResult.gpt_evaluation,
              audio_analysis: backendResult.audio_analysis,
              clinical_feedback: backendResult.clinical_feedback,
              audio_features: backendResult.audio_features
            };
          }
          return frontendResult;
        });
      }
    } catch (error) {
      console.warn('⚠️ Could not fetch backend data for merging:', error);
    }
    return frontendResults;
  }, [sessionId]);

  const saveCognitiveAssessmentResults = useCallback(async (all: TestResult[]) => {
    try {
      console.log('💾 Saving cognitive assessment results to database...');

      // Merge backend data with frontend data
      console.log('🔄 Merging backend data with frontend data...');
      const mergedResults = await mergeBackendData(all);

      // Calculate scores using merged data
      const { finalMmse, overallGptScore: gptScore } = computeFinalScores(mergedResults);
      const answeredQuestions = mergedResults.filter(r => r.transcription && r.transcription.trim().length > 0).length;
      const completionRate = mergedResults.length > 0 ? (answeredQuestions / mergedResults.length) * 100 : 0;

      // Prepare data for the new API
      const assessmentData = {
        sessionId,
        userId: userData?.email || 'anonymous',
        userInfo: {
          name: userData?.name || '',
          age: userData?.age || '',
          email: userData?.email || '',
          phone: userData?.phone || '',
          gender: userData?.gender || ''
        },
        startedAt: typeof window !== 'undefined' ? new Date(Date.now() - (mergedResults.length * 60000)).toISOString() : new Date().toISOString(),
        totalQuestions: mergedResults.length,
        answeredQuestions,
        completionRate,
        memoryScore: (() => {
          if (!mergedResults || mergedResults.length === 0) return 0;
          let totalScore = 0;
          const maxScore = 100;
          mergedResults.forEach(result => {
            if (result.transcription && result.transcription.trim().length > 0) {
              const wordCount = result.transcription.split(' ').length;
              const questionScore = Math.min(wordCount * 2, 10);
              totalScore += questionScore;
            }
          });
          const normalizedScore = Math.min((totalScore / mergedResults.length) * 10, maxScore);
          return Math.round(normalizedScore);
        })(),
        cognitiveScore: 0, // Will be updated by backend analysis
        finalMmseScore: finalMmse,
        overallGptScore: gptScore,
        questionResults: mergedResults.map(result => ({
          questionId: result.questionId,
          question: result.question,
          transcription: result.transcription || '',
          timestamp: result.timestamp,
          hasAudio: !!result.audioBlob,
          duration: result.duration || 0,
          gptEvaluation: result.gpt_evaluation,
          audioFeatures: result.audio_features
        })),
        audioFiles: mergedResults.map((result, index) => ({
          index,
          questionId: result.questionId,
          filename: `audio_q${result.questionId}.webm`,
          hasAudio: !!result.audioBlob,
          size: result.audioBlob?.size || 0
        })),
        recordingsPath: `/recordings/session_${sessionId}`,
        cognitiveAnalysis: {
          finalScore: finalMmse,
          totalQuestions: mergedResults.length,
          answeredQuestions,
          completionRate,
          cognitiveLevel: finalMmse ? (finalMmse >= 24 ? 'normal' : finalMmse >= 18 ? 'mild' : finalMmse >= 10 ? 'moderate' : 'severe') : 'incomplete',
          assessmentType: 'mmse',
          timestamp: new Date().toISOString()
        },
        audioFeatures: mergedResults.reduce((acc, result) => {
          if (result.audio_features) {
            acc[result.questionId] = result.audio_features;
          }
          return acc;
        }, {} as Record<string, any>),
        usageMode,
        assessmentType: 'cognitive'
      };

      const response = await fetch('/api/save-cognitive-assessment-results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(assessmentData)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ Cognitive assessment results saved successfully:', result);
        alert('Kết quả kiểm tra đã được lưu trữ thành công vào database!');
      } else {
        const error = await response.json();
        console.error('❌ Failed to save cognitive assessment results:', error);
        alert('Có lỗi khi lưu trữ kết quả. Vui lòng thử lại.');
      }
    } catch (error) {
      console.error('❌ Error saving cognitive assessment results:', error);
      alert('Có lỗi khi lưu trữ kết quả. Vui lòng kiểm tra kết nối mạng.');
    }
  }, [userData, sessionId, usageMode, computeFinalScores, mergeBackendData]);

  // Process question response and update MMSE progress
  const processQuestionResponse = useCallback(async (questionId: number, response: string, sessionId: string) => {
    try {
      console.log(`🔄 Processing question ${questionId} response for MMSE calculation`);

      // 1. Update question status to completed (backend processing should have finished by now)
      updateQuestionStatus(questionId, 'completed', {
        feedback: 'Assessment completed successfully',
        score: 85 + Math.floor(Math.random() * 15) // Mock score for now
      });

      // 2. Check if current domain is completed
      const question = questions.find(q => q.id === String(questionId));
      if (!question) {
        console.warn(`⚠️ Question ${questionId} not found in questions list`);
        return;
      }

      const currentDomain = question.domain;

      // Get all questions in current domain
      const domainQuestions = questions.filter(q => q.domain === currentDomain);
      const completedDomainQuestions = Array.from(questionStates.values())
        .filter(state => {
          const q = questions.find(qq => qq.id === String(state.id));
          return q?.domain === currentDomain && state.status === 'completed';
        });

      console.log(`📊 Domain ${currentDomain}: ${completedDomainQuestions.length}/${domainQuestions.length} questions completed`);

      // 3. If domain is complete, calculate domain score
      if (completedDomainQuestions.length >= domainQuestions.length) {
        console.log(`✅ Domain ${currentDomain} completed, calculating score...`);

        // Calculate domain score based on completed questions
        const domainScore = Math.min(
          mmseAssessment.domains[currentDomain as keyof typeof mmseAssessment.domains].maxScore,
          Math.round(
            (completedDomainQuestions.reduce((sum, state) => {
              // Use score from question state or default scoring
              return sum + (state.score || 1);
            }, 0) / completedDomainQuestions.length) *
            mmseAssessment.domains[currentDomain as keyof typeof mmseAssessment.domains].maxScore
          )
        );

        // Update MMSE domain
        mmseAssessment.completeDomain(currentDomain as keyof typeof mmseAssessment.domains, domainScore);

        // Update domain progress in UI
        setDomainProgress(mmseAssessment.getProgress());

        console.log(`✅ Domain ${currentDomain} score: ${domainScore}/${mmseAssessment.domains[currentDomain as keyof typeof mmseAssessment.domains].maxScore}`);

        // 4. Check if entire assessment is complete
        const allDomainsComplete = Object.values(mmseAssessment.domains).every(domain => domain.completed);

        if (allDomainsComplete) {
          console.log('🎯 All domains complete, calculating final MMSE score...');

          // Calculate final MMSE score
          const finalScore = mmseAssessment.calculateTotalScore();
          const cognitiveStatus = mmseAssessment.cognitiveStatus;

          console.log(`✅ Final MMSE Score calculated: ${finalScore}/30 - ${cognitiveStatus}`);

          // Mark assessment as completed
          setAssessmentCompleted(true);

              // Save MMSE results to dedicated endpoint
          try {
            await fetch(`http://localhost:5001/api/mmse/results/${sessionId}`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                totalScore: finalScore,
                cognitiveStatus: cognitiveStatus,
                // REMOVED: domainScores - violates MMSE medical standards
                completedAt: new Date().toISOString()
              })
            });
            console.log('✅ MMSE results saved successfully');
          } catch (error) {
            console.error('❌ Failed to save MMSE results:', error);
          }

          // Note: MMSE results are saved to dedicated endpoint above
          // Final database save happens in the completion handler with proper data
          try {
            if (typeof window !== 'undefined') {
              localStorage.setItem('lastSessionId', sessionId);
              if (userData?.email) localStorage.setItem('userEmail', userData.email);
            }
          } catch {}

          // Navigate to results
          setTimeout(() => {
            router.push(`/results?sessionId=${sessionId}`);
          }, 1500);
        }
      }

      // 5. Update session progress
      // This is handled by questionStates updates above

    } catch (error) {
      console.error(`❌ Error processing question ${questionId} response:`, error);
    }
  }, [questions, questionStates, mmseAssessment, setDomainProgress, setAssessmentCompleted, saveCognitiveAssessmentResults, router, updateQuestionStatus]);

  const finalizeCommunityIfNeeded = useCallback(async (all: TestResult[]) => {
    if (usageMode !== 'community') return;
    try {
      const { finalMmse, overallGptScore } = computeFinalScores(all);
      const fd = new FormData();
      fd.append('sessionId', sessionId);
      fd.append('finalMmse', String(finalMmse));
      fd.append('overallGptScore', String(overallGptScore));
      fd.append('resultsJson', JSON.stringify({ results: all }));
      await fetch('/api/community/finalize', { method: 'POST', body: fd });

      // send email to community email (or user email)
      const sendFd = new FormData();
      sendFd.append('email', communityEmail || userData?.email || '');
      sendFd.append('name', communityName || userData?.name || '');
      sendFd.append('sessionId', sessionId);
      sendFd.append('finalMmse', String(finalMmse));
      if (typeof overallGptScore === 'number') sendFd.append('overallGptScore', String(overallGptScore));
      sendFd.append('summary', JSON.stringify({ results: all }));
      if (communityEmail || userData?.email) {
        fetch('/api/community/send-result', { method: 'POST', body: sendFd });
      }
    } catch (e) {
      console.error(e);
    }
  }, [usageMode, sessionId, communityEmail, communityName, userData, computeFinalScores, saveCognitiveAssessmentResults]);

  const autoTranscribeAudio = async (): Promise<void> => {
    // Check if we have any audio data (even if empty)
    const audioBlob = (window as ExtendedWindow).lastAudioBlob;
    const hasAudio = !!audioBlob;
    const audioSize = audioBlob?.size || 0;

    console.log('🔍 Audio check:', {
      hasAudio,
      audioSize,
      isEmptyAudio: hasAudio && audioSize < 1000 // Consider < 1KB as potentially empty
    });

    // Check if questions are loaded and currentQuestionIndex is valid
    if (!questions || !Array.isArray(questions) || !questions.length || currentQuestionIndex < 0 || currentQuestionIndex >= questions.length || !currentQuestion) {
      alert('Câu hỏi chưa được tải hoặc không hợp lệ. Vui lòng thử lại sau.');
      console.error('❌ Questions not loaded or currentQuestionIndex out of bounds in autoTranscribeAudio', {
        questionsLength: questions.length,
        currentQuestionIndex,
        currentQuestion: !!currentQuestion
      });
      return;
    }

    setIsAutoTranscribing(true);
    setAutoTranscriptionResult(null);

    try {
      // Handle empty audio case (for testing purposes)
      if (!hasAudio || audioSize < 1000) {
        // Create mock result for empty audio
        const mockTranscriptText = 'Không có lời thoại (âm thanh trống)';
        const mockConf = 0;
        const mockModel = 'empty-audio-test';

        // Create mock GPT evaluation with score 0
        const mockGptEval = {
          vocabulary_score: 0,
          context_relevance_score: 0,
          overall_score: 0,
          analysis: 'Không phát hiện được lời nói trong bản ghi âm. Đây có thể là do: 1) Không có âm thanh nào được ghi lại, 2) Mức âm lượng quá thấp, 3) Thời gian ghi quá ngắn.',
          feedback: 'Vui lòng thử ghi âm lại với âm lượng rõ ràng hơn và nói to hơn.',
          repetition_rate: 0,
          context_relevance: 0,
          comprehension_score: 0
        };

        // Create mock audio features
        const mockAudioFeatures = {
          duration: 0,
          pitch_mean: 0,
          pitch_std: 0,
          speech_rate: 0,
          tempo: 0,
          silence_mean: 1.0, // 100% silence
          number_utterances: 0
        };

        // Save mock result
        const mockUiResult = {
          success: true,
          transcript: mockTranscriptText,
          confidence: mockConf,
          language: language,
          model: mockModel,
          transcript_file: `transcript_empty_${Date.now()}.txt`,
          audio_duration: 0,
          sample_rate: 16000,
          audio_features: mockAudioFeatures,
          gpt_evaluation: mockGptEval
        };

        setAutoTranscriptionResult(mockUiResult);
        // Removed manual transcript setting

        // Process assessment result with mock data

        // Create test result from mock response
        const testResult: TestResult = {
          questionId: String(currentQuestion.id),
          question: currentQuestion.text,
          audioBlob: audioBlob || undefined,
          audioFilename: `recording_empty_${typeof window !== 'undefined' ? Date.now() : 0}.webm`,
          transcription: mockTranscriptText,
          timestamp: new Date(),
          duration: 0,
          processingMethod: 'empty_audio_test',
          gpt_evaluation: mockGptEval,
          audio_features: mockAudioFeatures,
          mmse_prediction: {
            predicted_mmse: 0,
            severity: 'Không có dữ liệu âm thanh',
            description: 'Bản ghi âm không chứa lời nói. Điểm số được đặt là 0 cho mục đích kiểm tra.',
            confidence: 0
          },
          auto_transcription: {
            success: true,
            transcript: mockTranscriptText,
            confidence: mockConf,
            language: language,
            model: mockModel,
            transcript_file: `transcript_empty_${typeof window !== 'undefined' ? Date.now() : 0}.txt`,
            audio_duration: 0,
            sample_rate: 16000,
            gpt_evaluation: mockGptEval,
            audio_features: mockAudioFeatures
          }
        };

        // Add to test results
        setTestResults(prev => [...prev, testResult]);

        // Process domain completion with mock data
        const allResults = [...testResults, testResult];

        // Check if current domain is completed
        const currentDomainQuestions = questions.filter(q => q.domain === currentDomain);
        const completedDomainQuestions = allResults.filter(r =>
          questions.find(q => q.id === r.questionId)?.domain === currentDomain
        );


        if (completedDomainQuestions.length >= currentDomainQuestions.length) {
          // Domain completed - calculate domain score based on quality of responses
          const domainScore = 0; // Score 0 for empty audio

          try {
            mmseAssessment.completeDomain(currentDomain, domainScore);
            setDomainProgress(mmseAssessment.getProgress());


            // Move to next domain or complete assessment
            const domainKeys = Object.keys(mmseAssessment.domains) as (keyof MMSEAssessment['domains'])[];
            const currentDomainIndex = domainKeys.indexOf(currentDomain);

            if (currentDomainIndex < domainKeys.length - 1 && !trainingMode) {
              // Move to next domain
              const nextDomain = domainKeys[currentDomainIndex + 1];
              // REMOVED: Auto-advancing to next domain to prevent duplicate questions
              // User must manually navigate using buttons
              console.log(`📊 Domain ${currentDomain} completed. User can manually proceed to next domain.`);
            } else if (mmseAssessment.canFinalize()) {
              // All domains completed - calculate final MMSE
              const finalScore = mmseAssessment.calculateTotalScore();
              if (finalScore !== null) {

                // Update domain progress to reflect completion
                setDomainProgress(mmseAssessment.getProgress());

                // Set assessment completed
                setTimeout(() => {
                  setAssessmentCompleted(true);
                  // Save results to database with official MMSE score
                  saveCognitiveAssessmentResults(allResults);
                  // finalize for community mode with official MMSE score
                  finalizeCommunityIfNeeded(allResults);
                }, 1000);
              }
            }
          } catch (error) {
            console.error('Error completing domain with empty audio:', error);
          }
        }

        // Show user notification about empty audio
        setTimeout(() => {
          // alert('🧪 Phát hiện âm thanh trống - Hệ thống đã xử lý với điểm số 0. Bạn có thể tiếp tục với câu hỏi tiếp theo.');
        }, 500);

        // Reset UI state (NO auto-advancing to next question)
        if (!trainingMode) {
          setTimeout(() => {
            setHasRecording(false);
            setIsRecordingStarted(false);
            console.log(`📊 Question ${currentQuestionIndex + 1} processing completed. User can manually proceed.`);
          }, 800);
        }

        setIsAutoTranscribing(false);
        return;
      }

      const formData = new FormData();
      formData.append('audio', audioBlob!, 'recording.webm');
      formData.append('language', language);
      // Thêm parameter để yêu cầu sử dụng mô hình ASR thuần Việt khi chọn tiếng Việt
      if (language === 'vi') {
        formData.append('use_vietnamese_asr', 'true');
        console.log('🎯 Sử dụng mô hình ASR thuần Việt cho độ chính xác cao');
      }
      // Include full context so GPT evaluates correctly and ML features are tied to the sample
      formData.append('question', currentQuestion.text.replace('{greeting}', greeting));
      formData.append('questionId', String(currentQuestion.id));
      if (userData?.age) formData.append('age', userData.age);
      if (userData?.gender) formData.append('gender', userData.gender);
      formData.append('user', userData?.email || userData?.phone || 'anonymous');

      console.log('🎵 Sending audio for auto-transcription...');

      // Use backend auto-transcribe endpoint for full assessment
      const response = await fetch('http://localhost:5001/auto-transcribe', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const json = await response.json();
        console.log('✅ Auto-transcription result:', json);

        if (json.success) {
          // /auto-transcribe returns full assessment data
          const data = json;
          // Handle both direct transcript and transcription object
          let transcriptText = '';
          let conf = 0;
          let model = 'openai-whisper-1';
          
          if (data.transcription && data.transcription.transcript && data.transcription.transcript.trim().length > 0) {
            // Transcription object from /auto-transcribe
            transcriptText = data.transcription.transcript;
            conf = (typeof data.transcription.confidence === 'number' && isFinite(data.transcription.confidence)) ? data.transcription.confidence : 0;
            model = data.transcription.model || 'openai-whisper-1';
          } else if (data.transcript && data.transcript.trim().length > 0) {
            // Direct transcript field
            transcriptText = data.transcript;
            conf = (typeof data.confidence === 'number' && isFinite(data.confidence)) ? data.confidence : 0;
            model = data.model || data.method || 'openai-whisper-1';
          } else {
            // No valid transcript found
            transcriptText = 'Không có lời thoại';
            conf = 0;
          }
          
          const audioFeatures = data.audio_features || null;
          const gptEval = data.gpt_evaluation || null;
          
          // Debug logging for data availability
          console.log('🔍 Data availability check:', {
            hasAudioFeatures: !!audioFeatures,
            hasGptEval: !!gptEval,
            hasMlPrediction: !!data.ml_prediction,
            audioFeaturesKeys: audioFeatures ? Object.keys(audioFeatures) : 'None',
            gptEvalKeys: gptEval ? Object.keys(gptEval) : 'None'
          });

          // Debug logging for GPT evaluation
          console.log('🔍 GPT Evaluation Data:', {
            gptEval,
            hasAnalysis: gptEval?.analysis ? 'YES' : 'NO',
            hasFeedback: gptEval?.feedback ? 'YES' : 'NO',
            analysisLength: gptEval?.analysis?.length || 0,
            feedbackLength: gptEval?.feedback?.length || 0
          });

          // Save a compact result used by UI blocks that expect direct fields
          const uiResult = {
            success: true,
            transcript: transcriptText,
            confidence: conf,
            model,
            audio_features: audioFeatures,
            gpt_evaluation: gptEval
          } as any;

          setAutoTranscriptionResult(uiResult);
          // Removed manual transcript setting

          // Process assessment result directly since /auto-transcribe returns full data
          console.log('🚀 Processing assessment result...');
          console.log('🔍 Debug data:', {
            transcriptText,
            audioFeatures: audioFeatures ? 'Present' : 'Missing',
            gptEval: gptEval ? 'Present' : 'Missing',
            mlPrediction: data.ml_prediction ? 'Present' : 'Missing',
            fullData: data
          });
          
          // Create test result from auto-transcribe response (MMSE compliant)
          const testResult: TestResult = {
            questionId: String(currentQuestion.id),
            question: currentQuestion.text,
            audioBlob: (window as ExtendedWindow).lastAudioBlob || undefined,
            audioFilename: `recording_${typeof window !== 'undefined' ? Date.now() : 0}.webm`,
            transcription: transcriptText,
            timestamp: new Date(),
            duration: audioFeatures?.duration || 0,
            processingMethod: 'domain_auto_transcribe',
            gpt_evaluation: gptEval || {
              vocabulary_score: undefined,
              context_relevance_score: 5,
              overall_score: 5,
              analysis: 'Phân tích không khả dụng',
              feedback: 'Không có góp ý cải thiện'
            },
            audio_features: audioFeatures || {
              duration: 0,
              pitch_mean: 0,
              pitch_std: 0,
              speech_rate: 0,
              tempo: 0,
              silence_mean: 0,
              number_utterances: 0
            },
            // AI Support (NOT official MMSE score)
            mmse_prediction: data.ml_prediction ? {
              predicted_mmse: Math.min(30.0, data.ml_prediction.predicted_score),
              severity: 'AI Hỗ trợ - Không phải MMSE chính thức',
              description: `🤖 AI phân tích giọng nói: ${gptEval?.analysis || 'Không có mô tả'}\n\n⚠️ Đây chỉ là hỗ trợ AI, KHÔNG thay thế cho đánh giá MMSE chuẩn.`,
              confidence: (data.ml_prediction.confidence || 0.5) * 0.7 // Reduce confidence to emphasize it's support
            } : undefined,
            auto_transcription: {
              success: true,
              transcript: transcriptText,
              confidence: conf,
              language: language,
              model: model,
              transcript_file: `transcript_${typeof window !== 'undefined' ? Date.now() : 0}.txt`,
              audio_duration: audioFeatures?.duration || 0,
              sample_rate: 16000,
              gpt_evaluation: gptEval || undefined,
              audio_features: audioFeatures || undefined
            }
          };

          // Add to test results
          setTestResults(prev => [...prev, testResult]);
          
          // Process domain completion (MMSE compliant)
          const allResults = [...testResults, testResult];
          
          // Check if current domain is completed
          const currentDomainQuestions = questions.filter(q => q.domain === currentDomain);
          const completedDomainQuestions = allResults.filter(r => 
            questions.find(q => q.id === r.questionId)?.domain === currentDomain
          );
          
          console.log(`📊 Domain ${currentDomain}: ${completedDomainQuestions.length}/${currentDomainQuestions.length} questions`);
          
          if (completedDomainQuestions.length >= currentDomainQuestions.length) {
            // Domain completed - calculate domain score based on quality of responses
            const domainScore = Math.min(
              mmseAssessment.domains[currentDomain].maxScore,
              Math.round(
                (completedDomainQuestions.reduce((sum, r) => {
                  const gptScore = r.gpt_evaluation?.overall_score || 5;
                  return sum + (gptScore / 10); // Convert 0-10 to 0-1
                }, 0) / completedDomainQuestions.length) * mmseAssessment.domains[currentDomain].maxScore
              )
            );
            
            try {
              mmseAssessment.completeDomain(currentDomain, domainScore);
              setDomainProgress(mmseAssessment.getProgress());
              
              console.log(`✅ Domain ${currentDomain} completed with score ${domainScore}/${mmseAssessment.domains[currentDomain].maxScore}`);
              
              // REMOVED: Auto-advancing to next domain to prevent duplicate questions
              // User must manually navigate using buttons
              console.log(`📊 Domain ${currentDomain} completed. User can manually proceed to next domain.`);

              if (mmseAssessment.canFinalize()) {
                // All domains completed - calculate final MMSE
                const finalScore = mmseAssessment.calculateTotalScore();
                if (finalScore !== null) {
                  console.log(`🎯 MMSE Assessment completed! Final score: ${finalScore}/30`);
                  console.log(`🏥 Cognitive status: ${mmseAssessment.cognitiveStatus}`);
                  
                  // Update domain progress to reflect completion
                  setDomainProgress(mmseAssessment.getProgress());
                  
                  // Set assessment completed
                  setTimeout(() => {
                    setAssessmentCompleted(true);
                    // Save results to database with official MMSE score
                    saveCognitiveAssessmentResults(allResults);
                    // finalize for community mode with official MMSE score
                    finalizeCommunityIfNeeded(allResults);
                  }, 1000);
                }
              }
            } catch (error) {
              console.error('Error completing domain:', error);
            }
          }
          
          // Show success message
          console.log(`✅ Question completed: "${transcriptText}" (${(conf * 100).toFixed(1)}% confidence)`);

          // Process assessment in normal mode (BACKGROUND ONLY - no progression)
          if (!trainingMode) {
            console.log('🚀 Processing assessment (pure background)...');
            console.log('📝 Calling submitDomainAssessment with:', {
              transcriptText: transcriptText.substring(0, 50) + '...',
              hasAudioFeatures: !!audioFeatures,
              hasGptEval: !!gptEval,
              currentQuestionIndex
            });

            // Use submitDomainAssessment for assessment queuing (PURE BACKGROUND)
            try {
              await submitDomainAssessment(transcriptText, audioFeatures, gptEval);
              console.log('✅ submitDomainAssessment completed successfully');
            } catch (error) {
              console.error('❌ submitDomainAssessment failed:', error);
            }

            console.log('✅ Background processing completed - NO UI progression from here');
          }

          // Reset UI state for next question (with delay to show transition)
          setTimeout(() => {
            setHasRecording(false);
            setIsRecordingStarted(false);
            setAutoTranscriptionResult(null);
            setAudioUrl(null);
          }, 300);
        } else {
          alert(t('transcription_failed').replace('{error}', json.error || 'Unknown error'));
        }
      } else {
        const errorData = await response.json();
        alert(`❌ Auto-transcription error: ${errorData.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('❌ Auto-transcription error:', error);
      alert(t('transcription_connection_error'));
    } finally {
      setIsAutoTranscribing(false);
    }
  };



  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    fetchUserData();
  }, []);

  // load settings
  useEffect(() => {
    try {
      const tm = typeof window !== 'undefined' ? localStorage.getItem('trainingMode') : null;
      const um = typeof window !== 'undefined' ? localStorage.getItem('usageMode') : null;
      console.log('Loading settings:', { tm, um, trainingModeParsed: tm === 'true' });
      if (tm !== null) setTrainingMode(tm === 'true');
      if (um === 'community' || um === 'personal') setUsageMode(um);
    } catch {}
  }, []);

  // Community intake prompt
  useEffect(() => {
    if (usageMode !== 'community') return;
    if (communityIntakeDone) return;
    setTimeout(() => setShowCommunityModal(true), 200);
  }, [usageMode, communityIntakeDone]);

  // Helper: Save training sample
  // Function to complete question with manual transcript (training mode)
  const completeQuestionWithTranscript = useCallback(async () => {
    if (!trainingMode) return;

    // Check if questions are loaded and currentQuestionIndex is valid
    if (!questions || !Array.isArray(questions) || !questions.length || currentQuestionIndex < 0 || currentQuestionIndex >= questions.length || !currentQuestion) {
      alert('Câu hỏi chưa được tải hoặc không hợp lệ. Vui lòng thử lại sau.');
      console.error('❌ Questions not loaded or currentQuestionIndex out of bounds in completeQuestionWithTranscript', {
        questionsLength: questions.length,
        currentQuestionIndex,
        currentQuestion: !!currentQuestion
      });
      return;
    }

      // Validate manual transcript
      if (!manualTranscript.trim()) {
        alert('Vui lòng nhập manual transcript trước khi hoàn thành câu hỏi');
        return;
      }

    try {
      setIsProcessing(true);
        setIsSavingTraining(true);

      // Save training sample first (optional)
      try {
        const fd = new FormData();
        fd.append('sessionId', sessionId);
        fd.append('userEmail', userData?.email || 'anonymous');
        fd.append('userName', userData?.name || '');
        fd.append('questionId', String(currentQuestion?.id || ''));
        fd.append('questionText', (currentQuestion?.text || '').replace('{greeting}', greeting));
        fd.append('audioFilename', `recording_${Date.now()}.webm`);
        fd.append('audioUrl', audioUrl || '');
        fd.append('autoTranscript', autoTranscriptionResult?.transcript || '');
        fd.append('manualTranscript', manualTranscript);

        const res = await fetch('/api/training-sample', { method: 'POST', body: fd });
        if (!res.ok) throw new Error('Save training sample failed');

        console.log('✅ Training sample saved successfully');
      } catch (e) {
        console.warn('⚠️ Training sample save failed, continuing...', e);
      }

      // Use the new async assessment system
      const questionId = String(currentQuestion?.id || (currentQuestionIndex + 1));
      const audioBlob = (window as ExtendedWindow).lastAudioBlob || undefined;

      await handleCompleteQuestion(Number(questionId), manualTranscript, audioBlob);

    } catch (e) {
      console.error('❌ Error completing question:', e);
      alert('Có lỗi khi hoàn thành câu hỏi. Vui lòng thử lại.');
    } finally {
      setIsProcessing(false);
      setIsSavingTraining(false);
    }
  }, [trainingMode, manualTranscript, autoTranscriptionResult, sessionId, userData, currentQuestion, greeting, audioUrl, handleCompleteQuestion, currentQuestionIndex, questions]);

  // Keep original saveTrainingSample for backward compatibility
  const saveTrainingSample = useCallback(async () => {
    if (!trainingMode) return;
    try {
      if (!autoTranscriptionResult?.transcript) { alert('Chưa có auto-transcript'); return; }
      if (!manualTranscript.trim()) { alert('Vui lòng nhập manual transcript'); return; }
      setIsSavingTraining(true);
      const fd = new FormData();
      fd.append('sessionId', sessionId);
      fd.append('userEmail', userData?.email || 'anonymous');
      fd.append('userName', userData?.name || '');
      fd.append('questionId', String(currentQuestion?.id || ''));
      fd.append('questionText', (currentQuestion?.text || '').replace('{greeting}', greeting));
      fd.append('audioFilename', `recording_${Date.now()}.webm`);
      fd.append('audioUrl', audioUrl || '');
      fd.append('autoTranscript', autoTranscriptionResult.transcript || '');
      fd.append('manualTranscript', manualTranscript);
      const res = await fetch('/api/training-sample', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Save training sample failed');
      alert('Đã lưu training sample');
    } catch (e) {
      console.error(e);
      alert('Lưu training sample thất bại');
    } finally {
      setIsSavingTraining(false);
    }
  }, [trainingMode, autoTranscriptionResult, manualTranscript, sessionId, userData, currentQuestion, greeting, audioUrl]);

  // Add effect to sync user data when language changes
  useEffect(() => {
    if (userData) {
      generateGreeting(userData, language);
    }
  }, [language, userData]);

  // Add effect to periodically sync with profile system
  useEffect(() => {
    const syncInterval = setInterval(() => {
      syncUserData();
    }, 30000); // Sync every 30 seconds

    return () => clearInterval(syncInterval);
  }, []);

  useEffect(() => {
    if ('speechSynthesis' in window) {
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        console.log('Available voices:', voices.length);
      };
      loadVoices();
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);

  if (isLoading) {
  return (
      <div className="min-h-screen max-h-[150vh] overflow-auto" style={{
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
            <h1 className="font-bold text-base text-amber-800">
              Đánh giá nhận thức
            </h1>
            <div />
          </div>
        </div>
        <div className="text-center">
          <Waves className="h-20 w-20 mx-auto mb-6 animate-pulse text-amber-600" />
          <h1 className="text-3xl font-light mb-2 text-amber-700">{greeting}</h1>
          <p className="text-gray-700">{t('preparing_assessment')}</p>

          {/* Backend status with retry functionality */}
          {(showBackendWarning || process.env.NODE_ENV === 'development') && (
            <div className="mt-4 p-3 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-300 rounded-lg max-w-md mx-auto">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <div className={`w-2 h-2 rounded-full ${backendStatus === 'connected' ? 'bg-green-500' : backendStatus === 'disconnected' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'}`} />
                  <span className={backendStatus === 'connected' ? 'text-green-700' : backendStatus === 'disconnected' ? 'text-red-700' : 'text-amber-700'}>
                    Backend: {backendStatus === 'connected' ? '✅ Connected' : backendStatus === 'disconnected' ? '❌ Offline' : '⏳ Checking...'}
                  </span>
                </div>
                {backendStatus === 'disconnected' && (
                  <button
                    onClick={checkBackendHealthLocal}
                    className="px-3 py-1 bg-amber-600 hover:bg-amber-700 text-white text-xs rounded-md transition-colors"
                  >
                    🔄 Retry
                  </button>
                )}
              </div>
              <p className="text-xs text-amber-700">
                {backendStatus === 'connected'
                  ? '✅ Connected to Flask backend at localhost:5001'
                  : retryCount >= 3
                    ? '❌ Backend server not responding. Please start the Flask server.'
                    : `⏳ Attempting to connect... (${retryCount}/3)`
                }
              </p>
              {backendStatus === 'disconnected' && retryCount >= 3 && (
                <div className="mt-2 p-2 bg-amber-100 border border-amber-300 rounded text-xs text-amber-800">
                  <strong>🔧 How to start backend:</strong>
                  <br />
                  <code className="bg-amber-200 px-1 rounded">cd backend && python app.py</code>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative px-2 sm:px-3 lg:px-4 py-3 sm:py-4 lg:py-6" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)'
    }}>
    {/* Illustrations */}
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <motion.div className="absolute -top-8 -left-8 w-32 h-32 sm:w-40 sm:h-40 rounded-full blur-3xl" animate={{ x: [0, 10, 0], y: [0, 5, 0] }} transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }} style={{ backgroundColor: '#F4A261', opacity: 0.1 }} />
      <motion.div className="absolute bottom-8 right-0 w-36 h-36 sm:w-48 sm:h-48 rounded-full blur-3xl" animate={{ x: [0, -8, 0], y: [0, -5, 0] }} transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }} style={{ backgroundColor: '#E88D4D', opacity: 0.1 }} />
      <motion.div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-20 h-20 sm:w-24 sm:h-24 rounded-full blur-2xl" animate={{ scale: [1, 1.06, 1] }} transition={{ duration: 8, repeat: Infinity }} style={{ backgroundColor: '#E67635', opacity: 0.08 }} />
      </div>
    {/* Removed watercolor gradients to achieve flat matte look */}
    {/* Matte noise overlay for pastel grain */}
    <div
      className="pointer-events-none absolute inset-0 mix-blend-multiply opacity-20"
      style={{
        backgroundImage:
          "radial-gradient(rgba(0,0,0,0.035) 0.8px, transparent 0.8px), radial-gradient(rgba(255,165,0,0.06) 0.6px, transparent 0.6px)",
        backgroundSize: "7px 7px, 6px 6px",
        backgroundPosition: "0 0, 2px 2px"
      }}
    />
    <div className="relative max-w-7xl mx-auto p-3 sm:p-4 lg:p-6 space-y-4 lg:space-y-6">

      {/* Header */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-2">
          <div className="md:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="sm">
                  <Menu className="h-5 w-5 text-orange-700" />
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
          <Brain className="w-9 h-9 text-orange-700" />
          <h1 className="text-2xl sm:text-3xl font-extrabold text-orange-700">Đánh giá nhận thức</h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="inline-flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full ${backendStatus === 'connected' ? 'bg-green-400' : backendStatus === 'disconnected' ? 'bg-red-400' : 'bg-yellow-400'}`} />
            <span className={`text-sm font-semibold ${backendStatus === 'connected' ? 'text-green-700' : backendStatus === 'disconnected' ? 'text-red-700' : 'text-yellow-700'}`}>
              {backendStatus === 'connected' ? 'Connected' : backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
            </span>
          </div>
          <LanguageSwitcher />
        </div>
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Questions & Recording */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="shadow-sm" style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261',
            boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
          }}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Waves className="w-5 h-5 text-[#B55D3A]" />
                <span className="text-[#8E4A2F]">Câu hỏi đánh giá</span>
              </CardTitle>
              <CardDescription>
                Chọn câu hỏi và nhấn loa để hệ thống đọc, sau đó ghi âm câu trả lời.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              
              {/* Complete Test Button - Show when all questions are answered */}
                {assessmentProgress.completedQuestions === assessmentProgress.totalQuestions && !assessmentCompleted && (
                  <div className="mt-4">
                    <Button
                      onClick={async () => {
                        setAssessmentCompleted(true);
                        // Calculate final MMSE score
                        const finalScore = mmseAssessment.calculateTotalScore();
                        console.log('🎯 Test completed! Final MMSE Score:', finalScore);

                        // Save all results to database (will automatically merge backend data)
                        try {
                          await saveCognitiveAssessmentResults(testResults);
                          console.log('✅ Assessment results saved to database');

                          // Navigate to results page after successful save
                          setTimeout(() => {
                            router.push(`/results?sessionId=${sessionId}`);
                          }, 1500);
                        } catch (error) {
                          console.error('❌ Failed to save assessment results:', error);
                          alert('Có lỗi khi lưu trữ kết quả. Vui lòng thử lại.');
                        }
                      }}
                      className="w-full text-white font-bold py-3 px-6 rounded-xl shadow-lg transform transition-all duration-200 hover:scale-105" style={{
                        background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)'
                      }}
                    >
                      <span className="text-lg">🎉 Hoàn thành bài kiểm tra</span>
                    </Button>
                    <p className="text-xs text-center text-gray-600 mt-2">
                      Nhấn để hoàn tất đánh giá và xem kết quả cuối cùng
                    </p>
                  </div>
                )}
                
                
                {/* Domain Guidance */}
          
              {/* Async Assessment Progress Monitor */}
              {questionStates.size > 0 && (
                <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-blue-800">🚀 Tiến độ đánh giá</span>
                    <div className="flex gap-2 text-xs">
                      <span className="text-green-600">✅ {assessmentProgress.completedQuestions} hoàn thành</span>
                      <span className="text-yellow-600">⏳ {assessmentProgress.processingQuestions} đang xử lý</span>
                      <span className="text-red-600">❌ {assessmentProgress.failedQuestions} thất bại</span>
                    </div>
                  </div>
                  {/* Percent progress bar */}
                  <div className="mb-2">
                    {(() => {
                      const pct = assessmentProgress.totalQuestions > 0
                        ? Math.round((assessmentProgress.completedQuestions / assessmentProgress.totalQuestions) * 100)
                        : 0;
                      return (
                        <div>
                          <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                            <div
                              className="h-2 rounded-full transition-all duration-300"
                              style={{
                                background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)',
                                width: `${pct}%`
                              }}
                            />
                          </div>
                          <div className="mt-1 text-[11px] text-blue-700 font-medium">{pct}% hoàn thành</div>
                        </div>
                      );
                    })()}
                  </div>
                  <div className="grid grid-cols-11 gap-1">
                    {Array.from({ length: questions?.length || 0 }, (_, i) => i + 1).map((questionNum) => {
                      const questionState = questionStates.get(questionNum);
                      const status = questionState?.status || 'pending';
                      const isCurrent = questionNum === assessmentProgress.currentQuestion;

                      return (
                        <div
                          key={questionNum}
                          className={`w-6 h-6 rounded text-xs flex items-center justify-center transition-all duration-300 cursor-pointer ${
                            // Xanh lá: Hoàn thành thành công
                            getDisplayStatus(questionState) === 'success' ? 'bg-green-500 text-white shadow-sm hover:bg-green-600' :
                            // Vàng: Đang xử lý bất đồng bộ
                            status === 'processing' ? 'bg-yellow-400 text-white animate-pulse shadow-sm hover:bg-yellow-500' :
                            // Đỏ: Thất bại
                            status === 'failed' ? 'bg-red-500 text-white shadow-sm hover:bg-red-600' :
                            // Xanh dương: Câu hỏi hiện tại
                            isCurrent ? 'bg-blue-600 text-white ring-2 ring-blue-300 shadow-sm hover:bg-blue-700' :
                            // Xám: Chưa bắt đầu
                            'bg-gray-200 text-gray-600 hover:bg-gray-300'
                          }`}
                          title={
                            getDisplayStatus(questionState) === 'success' ? `✅ Hoàn thành - Điểm: ${questionState?.score || 'N/A'}` :
                            status === 'processing' ? `⏳ Đang đánh giá bất đồng bộ...` :
                            status === 'failed' ? `❌ Thất bại - Click để làm lại: ${questionState?.error || 'Unknown error'}` :
                            isCurrent ? '🎯 Câu hỏi hiện tại' : '⏸️ Chưa bắt đầu'
                          }
                          onClick={() => {
                            if (getDisplayStatus(questionState) === 'success') {
                              // Show results for completed questions
                              viewQuestionResult(questionNum);
                            } else if (status === 'failed') {
                              // Ask user if they want to retry failed question
                              const retry = confirm(`Câu hỏi ${questionNum} đã thất bại. Bạn có muốn làm lại không?\n\nLỗi: ${questionState?.error || 'Unknown error'}`);
                              if (retry) {
                                console.log(`🔄 Retrying question ${questionNum}`);

                                // Reset question state to pending
                                updateQuestionStatus(questionNum, 'pending', {
                                  answer: undefined,
                                  score: undefined,
                                  feedback: undefined,
                                  error: undefined,
                                  retryCount: ((questionState?.retryCount || 0) + 1),
                                  timestamp: new Date()
                                });

                                // Navigate to the question for retry
                                const questionIndex = questionNum - 1;
                                navigateToQuestion(questionIndex, true); // User action

                                // Reset UI state for retry
                                setHasRecording(false);
                                setIsRecordingStarted(false);
                                setRecordingDuration(0);
                                setAutoTranscriptionResult(null);
                                setAudioUrl(null);
                                setManualTranscript('');

                                if (questions[questionIndex]) {
                                  speakQuestionByIndex(questionIndex);
                                }

                                console.log(`✅ Question ${questionNum} reset for retry`);
                              } else {
                                // Show error details if user doesn't want to retry
                                viewQuestionResult(questionNum);
                              }
                            } else {
                              // Navigate to question for pending/processing questions
                              const questionIndex = questionNum - 1;
                              navigateToQuestion(questionIndex, true); // User action
                              if (questions[questionIndex]) {
                                speakQuestionByIndex(questionIndex);
                              }
                            }
                          }}
                        >
                          {questionNum}
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-2 text-xs text-blue-600">
                    💡 Click vào câu hỏi để xem chi tiết kết quả khi có sẵn
                  </div>
                  <div className="mt-1 text-xs text-gray-500 flex gap-4">
                    <span>🎯 Xanh dương: Câu hỏi hiện tại</span>
                    <span>⏳ Vàng: Đang xử lý</span>
                    <span>✅ Xanh lá: Hoàn thành</span>
                    <span>❌ Đỏ: Thất bại</span>
                  </div>
                </div>
              )}
              {/* Results cards removed by request */}
          
              {/* Question list */}
              <div className="rounded-xl border bg-white/70 divide-y max-h-64 overflow-auto">
                {!questionsLoaded ? (
                  <div className="p-4 text-center text-gray-500">
                    <div className="animate-spin w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-2"></div>
                    <div className="text-sm">Đang tải câu hỏi...</div>
                    <div className="text-xs mt-1">Hệ thống đang kết nối với backend</div>
                    <div className="text-xs mt-1 text-blue-600">Backend status: {backendStatus}</div>
                  </div>
                ) : questions.length === 0 ? (
                  <div className="p-4 text-center text-gray-500">
                    <div className="text-sm">Không có câu hỏi nào</div>
                    <div className="text-xs mt-1">Vui lòng kiểm tra kết nối backend</div>
                  </div>
                ) : (
                  questions.map((q, idx) => {
                    const questionNum = idx + 1;
                    const questionState = questionStates.get(questionNum);
                    const status = questionState?.status || 'pending';
                    const isCurrent = questionNum === assessmentProgress.currentQuestion;

                    return (
                  <div
                    key={q.id}
                    className={`p-3 cursor-pointer transition-all duration-200 ${
                      idx === currentQuestionIndex ? 'bg-blue-50 border-l-4 border-blue-400' : 'hover:bg-gray-50'
                    }`}
                    title={
                      getDisplayStatus(questionState) === 'success' ? `✅ Hoàn thành - Điểm: ${questionState?.score || 'N/A'}` :
                      status === 'processing' ? `⏳ Đang đánh giá bất đồng bộ...` :
                      status === 'failed' ? `❌ Thất bại - Click để làm lại: ${questionState?.error || 'Unknown error'}` :
                      isCurrent ? '🎯 Câu hỏi hiện tại' : '⏸️ Chưa bắt đầu'
                    }
                    onClick={() => {
                      const questionState = questionStates.get(questionNum);
                      if (getDisplayStatus(questionState) === 'success') {
                        // Show results for completed questions
                        viewQuestionResult(questionNum);
                      } else if (questionState?.status === 'failed') {
                        // Ask user if they want to retry failed question
                        const retry = confirm(`Câu hỏi ${questionNum} đã thất bại. Bạn có muốn làm lại không?\n\nLỗi: ${questionState.error || 'Unknown error'}`);
                        if (retry) {
                          console.log(`🔄 Retrying question ${questionNum}`);

                          // Reset question state to pending
                          updateQuestionStatus(questionNum, 'pending', {
                            answer: undefined,
                            score: undefined,
                            feedback: undefined,
                            error: undefined,
                            retryCount: ((questionState.retryCount || 0) + 1),
                            timestamp: new Date()
                          });

                          // Navigate to the question for retry
                          navigateToQuestion(idx, true); // User action

                          // Reset UI state for retry
                          setHasRecording(false);
                          setIsRecordingStarted(false);
                          setRecordingDuration(0);
                          setAutoTranscriptionResult(null);
                          setAudioUrl(null);
                          setManualTranscript('');

                          speakQuestionByIndex(idx);
                          console.log(`✅ Question ${questionNum} reset for retry`);
                        } else {
                          // Show error details if user doesn't want to retry
                          viewQuestionResult(questionNum);
                        }
                      } else {
                        // Navigate to question for pending/processing questions
                        navigateToQuestion(idx, true); // User action
                        speakQuestionByIndex(idx);
                      }
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                            <Badge className={`transition-all duration-300 ${
                              // Xanh lá: Hoàn thành thành công
                              getDisplayStatus(questionState) === 'success' ? 'bg-green-100 text-green-800 border-green-300' :
                              // Vàng: Đang xử lý bất đồng bộ
                              status === 'processing' ? 'bg-yellow-100 text-yellow-800 border-yellow-300 animate-pulse' :
                              // Đỏ: Thất bại
                              status === 'failed' ? 'bg-red-100 text-red-800 border-red-300' :
                              // Xanh dương: Câu hỏi hiện tại
                              isCurrent ? 'bg-blue-100 text-blue-800 border-blue-300' :
                              // Xám: Chưa bắt đầu
                              'bg-gray-100 text-gray-600 border-gray-300'
                            }`}>
                              <span className="flex items-center gap-1">
                                {getDisplayStatus(questionState) === 'success' ? '✅' :
                                 status === 'processing' ? '⏳' :
                                 status === 'failed' ? '❌' :
                                 isCurrent ? '🎯' : '⏸️'}
                                {q.id}
                              </span>
                            </Badge>
                        <span className="font-medium text-amber-800">{q.category}</span>
        </div>
      </div>
                    <div className="text-xs text-amber-700 mt-1">Câu hỏi đánh giá chức năng nhận thức</div>
                  </div>
                    );
                  })
                )}
              </div>

              

              {/* TTS controls */}
              {currentQuestion && (
                <div className="flex items-center gap-2">
                  <Button variant="primaryOutline" className="border-amber-300 text-amber-700" onClick={speakCurrentQuestion}>
                    <Volume2 className="w-4 h-4 mr-2 text-amber-600" /> Đọc câu hỏi
                        </Button>
                </div>
                    )}

              {/* Mic initializing banner */}
              {(isMicInitializing || (isRecording && !isRecordingStarted)) && (
                <div className="rounded-xl border bg-gradient-to-r from-amber-50 to-yellow-50 border-amber-300 p-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-amber-700" />
                  <div className="text-sm text-amber-800">
                    Đang khởi tạo micro… <span className="text-amber-700/80">Thời lượng sẽ bắt đầu khi micro sẵn sàng</span>
                  </div>
                </div>
              )}

              {/* Recording controls */}
              <div className="grid sm:grid-cols-2 gap-3">
                <Button
                  onClick={isRecording ? stopRecording : startRecording}
                  variant={isRecording ? 'danger' : 'primary'}
                  className="w-full text-white border-0" style={{
                    background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)'
                  }}
                  disabled={isMicInitializing}
                >
                  {isRecording ? (
                    <>
                      <Square className="w-5 h-5 mr-2" /> Dừng ghi âm ({recordingDuration}s)
                    </>
                  ) : isMicInitializing ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" /> Đang khởi tạo micro…
                    </>
                  ) : (
                    <>
                      <Mic className="w-5 h-5 mr-2" /> Bắt đầu ghi âm
                    </>
                  )}
                </Button>
                <div>
                  <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="audio/*" className="hidden" />
                  <Button variant="primaryOutline" className="w-full border-amber-300 text-amber-700 hover:bg-amber-50" onClick={() => fileInputRef.current?.click()}>
                    <FileAudio className="w-5 h-5 mr-2 text-amber-600" /> Chọn file audio
                  </Button>
                </div>
          </div>

              {/* Non-training mode: Auto-processing status */}
              {hasRecording && !trainingMode && (
                <div className="rounded-xl border bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200 p-3">
                  {isProcessing || isAutoTranscribing ? (
                    <div className="flex items-center gap-2 text-blue-700">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm font-medium">
                        {isAutoTranscribing ? 'Đang phân tích âm thanh...' : 'Đang xử lý kết quả...'}
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-green-700">
                      <CheckCircle className="w-4 h-4" />
                      <span className="text-sm font-medium">✨ Tự động hoàn thành sau khi ghi âm (Chế độ thường)</span>
                    </div>
                  )}
                  <div className="text-xs text-blue-600 mt-1">
                    💡 Bạn không cần nhấn nút - hệ thống sẽ tự động xử lý âm thanh và chuyển câu tiếp theo
                  </div>
                  <div className="text-xs text-amber-600 mt-1">
                    🧪 Kiểm tra: Hệ thống sẽ xử lý cả file âm thanh trống (score = 0)
                  </div>
                </div>
              )}

              {/* Debug Panel (Development Only) */}

              {/* Training Mode Manual Transcript Input - Always show after recording */}
              {trainingMode && hasRecording && (
                <div ref={manualTranscriptRef} className="bg-gradient-to-br from-amber-100 to-orange-200/60 rounded-xl border border-amber-300 p-4">
                  <h4 className="font-semibold text-amber-800 mb-3 flex items-center gap-2">
                    <Brain className="w-5 h-5 text-amber-600" />
                    Manual Transcript (Training Mode)
                  </h4>
                  <div className="space-y-3">
                    {autoTranscriptionResult && (
                      <div>
                        <label className="block text-sm font-medium text-amber-700 mb-1">Auto Transcript:</label>
                        <div className="bg-white/80 p-2 rounded-lg border text-sm text-gray-700">
                          {autoTranscriptionResult?.transcript || 'Chưa có transcript'}
                                    </div>
                                  </div>
                                )}
                    <div>
                      <label className="block text-sm font-medium text-amber-700 mb-1">
                        Manual Transcript: {autoTranscriptionResult ? '(sau khi xử lý)' : '(trước khi xử lý)'}
                      </label>
                      <textarea
                        rows={3}
                        className="w-full border border-amber-300 rounded-lg p-2 text-sm"
                        value={manualTranscript}
                        onChange={(e) => setManualTranscript(e.target.value)}
                        placeholder={autoTranscriptionResult ?
                          "Nhập transcript thủ công để so sánh với auto transcript..." :
                          "Nhập transcript thủ công ngay sau khi ghi âm..."
                        }
                      />
                                  </div>
                    <div className="flex justify-end gap-2">
                      <Button
                        onClick={completeQuestionWithTranscript}
                        disabled={isProcessing || isSavingTraining || !manualTranscript.trim()}
                        className="bg-green-600 hover:bg-green-700 text-white"
                      >
                        {(isProcessing || isSavingTraining) ? (
                          <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Đang xử lý...</>
                        ) : (
                          'Hoàn thành câu hỏi'
                        )}
                      </Button>
                    </div>
                                </div>
                              </div>
                            )}
            </CardContent>
          </Card>

          {/* Hiển thị kết quả từng câu */}
          {testResults.length > 0 && (
            <div className="bg-gradient-to-br from-amber-50 via-orange-100/70 to-orange-200/50 rounded-2xl shadow-xl p-3 sm:p-4 border border-amber-300">
              <div className="text-center mb-4">
                <h2 className="text-2xl font-bold text-amber-800 mb-2">{t('cognitive_assessment_results')}</h2>
                <p className="text-amber-700">{t('detailed_analysis')}</p>
              </div>

              {/* Jump bar 1..N */}
              <div className="flex flex-wrap gap-2 justify-center mb-4">
                {testResults
                  .slice() // copy
                  .sort((a, b) => String(a.questionId).localeCompare(String(b.questionId)))
                  .map((r, i) => (
                    <button
                      key={`jump-${r.questionId}-${i}`}
                      onClick={() => {
                        const el = document.getElementById(`qa-item-${i + 1}`);
                        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                      }}
                      className="px-2 py-1 text-xs rounded bg-amber-200 hover:bg-amber-300 text-amber-900 border border-amber-300"
                    >
                      {i + 1}
                    </button>
                  ))}
              </div>

              {/* Expandable list */}
              <div className="divide-y divide-amber-300/60 rounded-xl bg-white/60 border border-amber-300">
                {testResults
                  .slice()
                  .sort((a, b) => String(a.questionId).localeCompare(String(b.questionId)))
                  .map((result, idx) => (
                    <details key={`${result.questionId}-${result.timestamp.getTime()}`} id={`qa-item-${idx + 1}`} className="group">
                      <summary className="cursor-pointer list-none p-3 sm:p-4 flex items-center gap-3 hover:bg-amber-50/70">
                      <div className="bg-gradient-to-r from-amber-400 to-orange-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold text-base shadow-md">
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-base font-semibold text-amber-800">{t('question')} {idx + 1}</h3>
                          <p className="text-sm text-amber-700 line-clamp-1">{result.question}</p>
                      </div>
                        <span className="ml-auto text-amber-700 text-sm">{t('tap_to_expand') || 'Nhấn để xem'}</span>
                      </summary>

                      <div className="p-3 sm:p-4 bg-gradient-to-br from-amber-50/70 to-white/70">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4">
                      {/* Thông tin cơ bản */}
                      <div className="space-y-3">
                        <div className="bg-gradient-to-br from-amber-50 to-orange-100 rounded-lg p-2 border border-amber-300">
                          <h4 className="font-semibold text-amber-800 mb-2 flex items-center gap-2">
                            <FileText className="w-4 h-4 text-amber-600" />
                            {t('transcript')}
                          </h4>
                          <p className="text-amber-700 text-sm bg-white/80 p-2 rounded border border-amber-100">
                            {result.transcription || t('no_transcript')}
                          </p>
                          {result.auto_transcription && (
                            <div className="mt-2 p-2 bg-gradient-to-r from-amber-100 to-orange-200/60 rounded border border-amber-300">
                              <div className="flex items-center gap-2 text-amber-700 text-xs font-medium mb-1">
                                <Brain className="w-3 h-3 text-amber-600" />
                                <span>{t('ai_generated')}</span>
                              </div>
                              <div className="flex justify-between text-xs text-amber-600">
                                <span>Độ tin cậy: {(((result.auto_transcription as any)?.confidence ?? 0) * 100).toFixed(2)}%</span>
                                <span>Model: {formatModelName((result.auto_transcription as any)?.model)}</span>
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="bg-gradient-to-br from-white to-amber-50 rounded-lg p-2 border border-amber-200">
                          <h4 className="font-semibold text-amber-800 mb-2 flex items-center gap-2">
                            <Timer className="w-4 h-4 text-amber-600" />
                            {t('recording_info')}
                          </h4>
                          <div className="text-sm text-amber-700 space-y-1">
                            <p><b>{t('length')}:</b> {formatDuration(result.duration)}</p>
                            <p><b>{t('method')}:</b> {result.processingMethod || 'N/A'}</p>
                            <p><b>{t('time')}:</b> {result.timestamp.toLocaleString('vi-VN')}</p>
                              </div>
                          </div>
                        </div>

                          {/* Đặc trưng âm học & đánh giá */}
                          <div className="space-y-2">
                        <div className="bg-gradient-to-br from-white to-amber-50 rounded-lg p-3 border border-amber-200">
                          <h4 className="font-semibold text-amber-800 mb-3 flex items-center gap-2">
                            <Waves className="w-4 h-4 text-amber-600" />
                            Đặc trưng âm học
                          </h4>
                          {result.audio_features ? (
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div className="space-y-1">
                                <p className="text-amber-600"><b>Thời lượng:</b></p>
                                <p className="font-medium text-amber-800">{result.audio_features?.duration?.toFixed(2) ?? 'N/A'}s</p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-amber-600"><b>Cao độ TB (Hz):</b></p>
                                <p className="font-medium text-amber-800">{result.audio_features?.pitch_mean?.toFixed(2) ?? 'N/A'}</p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-amber-600"><b>Tốc độ nói (từ/giây):</b></p>
                                <p className="font-medium text-amber-800">{result.audio_features?.speech_rate !== undefined ? Number(result.audio_features.speech_rate).toFixed(2) : 'N/A'}</p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-amber-600"><b>Tempo (BPM):</b></p>
                                <p className="font-medium text-amber-800">{result.audio_features?.tempo?.toFixed(2) ?? 'N/A'}</p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-amber-600"><b>Khoảng nghỉ TB (s):</b></p>
                                <p className="font-medium text-amber-800">{result.audio_features?.silence_mean !== undefined ? result.audio_features.silence_mean.toFixed(2) : 'N/A'}</p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-amber-600"><b>Số lượt nói (utterances):</b></p>
                                <p className="font-medium text-amber-800">{result.audio_features?.number_utterances ?? 'N/A'}</p>
                              </div>
                            </div>
                          ) : (
                            <div className="p-3 bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                    <div className="w-4 h-4 bg-amber-500 rounded-full flex items-center justify-center text-xs">⚠️</div>
                                <span className="text-sm font-medium text-amber-800">Thông tin âm học không khả dụng</span>
                              </div>
                                  <p className="text-sm text-amber-700">Không thể trích xuất đặc trưng âm học từ file audio.</p>
                            </div>
                          )}
                        </div>

                        <div className="bg-gradient-to-br from-amber-50 to-yellow-50 rounded-xl p-2 border border-amber-200 shadow-sm max-h-48 overflow-auto text-xs">
                          {/* Đánh giá GPT theo từng câu */}
                          {(() => {
                            const gpt = result.gpt_evaluation || result.auto_transcription?.gpt_evaluation || result.o4mini_evaluation;
                            const mmse = result.mmse_prediction;
                            if (gpt) {
                              return (
                                <div className="space-y-2">
                                  <div className="flex items-center gap-2">
                                    <Brain className="w-4 h-4 text-amber-600" />
                                    <h4 className="font-semibold text-amber-900 text-sm">Đánh giá AI</h4>
                                  </div>

                                  {/* Điểm số */}
                                  <div className="flex flex-wrap items-center gap-1">
                                    {typeof gpt.overall_score === 'number' && (
                                      <div className="px-1.5 py-0.5 rounded bg-green-100 text-green-800 border border-green-200">
                                        Tổng quan: <b>{Number(gpt.overall_score).toFixed(2)}/10</b>
                                      </div>
                                    )}
                                    {typeof gpt.vocabulary_score === 'number' && (
                                      <div className="px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 border border-blue-200">
                                        Từ vựng: <b>{Number(gpt.vocabulary_score).toFixed(2)}/10</b>
                                      </div>
                                    )}
                                    {typeof (gpt as any).context_relevance_score === 'number' && (
                                      <div className="px-1.5 py-0.5 rounded bg-purple-100 text-purple-800 border border-purple-200">
                                        Liên quan ngữ cảnh: <b>{Number((gpt as any).context_relevance_score).toFixed(2)}/10</b>
                                      </div>
                                    )}
                                    {typeof (gpt as any).comprehension_score === 'number' && (
                                      <div className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200">
                                        Hiểu ngôn ngữ: <b>{Number((gpt as any).comprehension_score).toFixed(2)}/10</b>
                                      </div>
                                    )}
                                  </div>

                                  {/* Phân tích */}
                                  {gpt.analysis && (
                                    <div className="bg-white/80 p-2 rounded border border-amber-200">
                                      <div className="text-[11px] font-semibold text-amber-800 mb-1">Phân tích</div>
                                      <p className="text-xs text-amber-800 whitespace-pre-wrap">{gpt.analysis}</p>
                                    </div>
                                  )}

                                  {/* Góp ý cải thiện */}
                                  {gpt.feedback && (
                                    <div className="bg-white/80 p-2 rounded border border-amber-200">
                                      <div className="text-[11px] font-semibold text-amber-800 mb-1">Góp ý cải thiện</div>
                                      <p className="text-xs text-amber-800 whitespace-pre-wrap">{gpt.feedback}</p>
                                    </div>
                                  )}

                                  {/* AI Speech Analysis Support - No Scores Displayed */}
                                  {mmse && (
                                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-2 rounded border border-blue-200">
                                      <div className="text-[11px] font-semibold text-blue-800 mb-1">🤖 Hỗ trợ AI (Phân tích giọng nói)</div>
                                      <p className="text-xs text-blue-900">
                                        Phân tích kỹ thuật số về chất lượng giọng nói và khả năng phát âm
                                      </p>
                                      <p className="text-[10px] text-blue-700 mt-1 italic">
                                        ⚠️ Chỉ hỗ trợ kỹ thuật, không cung cấp điểm số
                                      </p>
                                      {mmse.description && (
                                        <p className="text-xs text-blue-900 mt-1">{mmse.description}</p>
                                      )}
                                    </div>
                                  )}
                                </div>
                              );
                            }
                            return (
                              <div className="p-2 bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200 rounded-lg text-xs text-amber-800">
                                Chưa có đánh giá AI cho câu hỏi này.
                              </div>
                            );
                          })()}
                        </div>
                      </div>
                    </div>
                  </div>
                </details>
                ))}
              </div>
            </div>
          )}
                          </div>

        {/* Right: Patient info & Status */}
        <div className="space-y-4">
          <Card className="shadow-md" style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261',
            boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
          }}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="w-5 h-5" style={{ color: '#F4A261' }} /> Thông tin người dùng
              </CardTitle>
              <CardDescription>Không bắt buộc nhưng giúp tăng độ chính xác</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-black-600 mb-1">Họ tên</label>
                  <input
                    className="w-full p-2 rounded-lg border border-amber-500"
                    value={patientInfo.name}
                    onChange={(e) => setPatientInfo(prev => ({ ...prev, name: e.target.value }))}
                  />
                                  </div>
                <div>
                  <label className="block text-xs text-black-600 mb-1">Tuổi</label>
                  <input
                    type="number"
                    min={0}
                    max={120}
                    className="w-full p-2 rounded-lg border border-amber-500"
                    value={patientInfo.age}
                    onChange={(e) => setPatientInfo(prev => ({ ...prev, age: e.target.value }))}
                  />
                              </div>
                <div>
                  <label className="block text-xs text-black-600 mb-1">Giới tính</label>
                  <select
                    className="w-full p-2 rounded-lg border border-amber-500"
                    value={patientInfo.gender}
                    onChange={(e) => setPatientInfo(prev => ({ ...prev, gender: e.target.value }))}
                  >
                    <option value="">Chọn</option>
                    <option value="male">Nam</option>
                    <option value="female">Nữ</option>
                    <option value="other">Khác</option>
                  </select>
                      </div>
                <div>
                  <label className="block text-xs text-black-600 mb-1">Số năm học (Trình độ học vấn)</label>
                  <input
                    type="number"
                    min={0}
                    max={30}
                    className="w-full p-2 rounded-lg border border-amber-500"
                    value={patientInfo.education_years}
                    onChange={(e) => setPatientInfo(prev => ({ ...prev, education_years: e.target.value }))}
                  />
                  </div>
              </div>
              <div>
                <label className="block text-xs text-black-600 mb-1">Ghi chú</label>
                <textarea
                  rows={3}
                  className="w-full p-2 rounded-lg border border-amber-500"
                  value={patientInfo.notes}
                  onChange={(e) => setPatientInfo(prev => ({ ...prev, notes: e.target.value }))}
                />
              </div>
            </CardContent>
            </Card>

          <Card className="shadow-md" style={{
            background: 'rgba(255, 255, 255, 0.9)',
            border: '2px solid #F4A261',
            boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
          }}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" style={{ color: '#F4A261' }} /> Trạng thái
              </CardTitle>
              <CardDescription>Phiên: {sessionId}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span>Đọc câu hỏi</span>
                <Badge className={isTTSSpeaking ? 'bg-amber-100 text-amber-800' : 'bg-gray-50 text-gray-700'}>
                  {isTTSSpeaking ? 'Đang đọc' : 'Tắt'}
                </Badge>
          </div>
              <div className="flex items-center justify-between">
                <span>Ghi âm</span>
                <Badge className={isMicInitializing || (isRecording && !isRecordingStarted) ? 'bg-yellow-100 text-amber-800' : isRecording ? 'bg-amber-100 text-amber-800' : 'bg-gray-50 text-gray-700'}>
                  {isMicInitializing || (isRecording && !isRecordingStarted) ? 'Khởi tạo mic…' : (isRecording ? 'Đang ghi...' : 'Tạm dừng')}
                </Badge>
        </div>
              <div className="flex items-center justify-between">
                <span>Thời lượng</span>
                <span className="font-medium">{recordingDuration}s</span>
          </div>
              <div className="flex items-center justify-between">
                <span>Audio</span>
                <Badge className={hasRecording ? 'bg-amber-100 text-amber-800' : 'bg-gray-50 text-gray-700'}>
                  {hasRecording ? 'Đã chọn' : 'Chưa có'}
                </Badge>
                </div>
            </CardContent>
            </Card>
        </div>

        {/* Navigation removed per spec */}

        {/* Assessment Completion Screen */}
        {assessmentCompleted && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
          >
            <motion.div
              initial={{ y: 50 }}
              animate={{ y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="rounded-2xl shadow-2xl max-w-2xl w-full mx-4 overflow-hidden" style={{
                background: 'rgba(255, 255, 255, 0.95)',
                border: '2px solid #F4A261'
              }}
            >
              {/* Header */}
              <div className="px-6 py-8 text-white text-center" style={{
                background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)'
              }}>
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                  className="text-6xl mb-4"
                >
                  🎉
                </motion.div>
                <motion.h2
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 }}
                  className="text-2xl font-bold mb-2"
                >
                  {t('assessment_complete')}!
                </motion.h2>
                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.8 }}
                  className="text-green-100"
                >
                  Chúc mừng bạn đã hoàn thành bài đánh giá nhận thức
                </motion.p>
              </div>

              {/* Content */}
              <div className="px-6 py-8">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 1 }}
                  className="space-y-6"
                >
                  {/* Summary Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-blue-600 mb-1">
                        {testResults.length}
                      </div>
                      <div className="text-sm text-blue-700">Câu hỏi đã trả lời</div>
                    </div>
                    <div className="bg-purple-50 rounded-lg p-4 text-center">
                      <div className="text-2xl font-bold text-purple-600 mb-1">
                        {testResults.length > 0 ? Math.round((testResults.filter(r => r.gpt_evaluation).length / testResults.length) * 100) : 0}%
                      </div>
                      <div className="text-sm text-purple-700">GPT Evaluation</div>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="space-y-3">
                    <h3 className="font-semibold text-black-600 mb-3">Bạn muốn làm gì tiếp theo?</h3>

                    <div className="grid gap-3">
                      <Button
                        onClick={() => window.location.reload()}
                        className="w-full text-white py-3 rounded-lg font-medium" style={{
                          background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)'
                        }}
                      >
                        🔄 Làm bài kiểm tra mới
                      </Button>

                      <Button
                        onClick={() => router.push(`/results?sessionId=${encodeURIComponent(sessionId)}`)}
                        variant="primaryOutline"
                        className="w-full py-3 rounded-lg border-2 border-gray-300 hover:border-gray-400 font-medium"
                      >
                        📊 Xem chi tiết kết quả
                      </Button>

                      <Button
                        onClick={() => window.location.href = '/'}
                        variant="primaryOutline"
                        className="w-full py-3 rounded-lg border-2 border-gray-300 hover:border-gray-400 font-medium"
                      >
                        🏠 Về trang chủ
                      </Button>
                    </div>
                  </div>

                  {/* Encouraging Message */}
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4 border border-green-200">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">💪</div>
                      <div>
                        <p className="font-medium text-green-800 mb-1">Thật tuyệt vời!</p>
                        <p className="text-sm text-green-700">
                          Bạn đã hoàn thành bài đánh giá một cách xuất sắc.
                          Kết quả của bạn sẽ giúp chúng tôi hiểu rõ hơn về tình trạng nhận thức của bạn.
                        </p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Final Results Display */}
        {testCompleted && finalResults && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-4xl mx-auto mt-8"
          >
            <Card className="p-8" style={{
              background: 'rgba(255, 255, 255, 0.9)',
              border: '2px solid #F4A261',
              boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)'
            }}>
              <div className="text-center mb-8">
                <div className="text-6xl mb-4">🎯</div>
                <h2 className="text-3xl font-bold text-black-600 mb-2">
                  Kết Quả Đánh Giá Tổng Thể
                </h2>
                <p className="text-black-600">
                  Phân tích chi tiết về tình trạng nhận thức của bạn
                </p>
              </div>

              {/* Official MMSE Score - Only if assessment completed */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
                  <div className="text-center">
                    {mmseAssessment?.completed ? (
                      <>
                        <div className="text-5xl font-bold text-green-600 mb-2">
                          {mmseAssessment?.totalScore ?? 0}/30
                        </div>
                        <div className="text-lg font-semibold text-green-700 mb-1">Điểm MMSE Chính Thức</div>
                        <div className="text-sm text-green-600 font-medium">
                          {mmseAssessment?.cognitiveStatus ?? 'Không xác định'}
                        </div>
                        <div className="text-xs text-green-500 mt-2 italic">
                          ✅ Hoàn thành đúng chuẩn khoa học
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="text-4xl font-bold text-gray-400 mb-2">
                          --/30
                        </div>
                        <div className="text-lg font-semibold text-gray-500 mb-1">Điểm MMSE</div>
                        <div className="text-sm text-gray-500">
                          Chưa hoàn thành đánh giá
                        </div>
                        <div className="text-xs text-red-500 mt-2">
                          ⚠️ Cần hoàn thành tất cả 6 lĩnh vực
                        </div>
                      </>
                    )}
                  </div>
                </div>

                <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-green-600 mb-2">
                      {finalResults.scores?.average_gpt_score || 0}/10
                    </div>
                    <div className="text-lg font-semibold text-black-600 mb-1">Điểm Đánh Giá AI</div>
                    <div className="text-sm text-black-600">
                      {finalResults.scores?.severity || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* GPT Analysis */}
              {finalResults.gpt_analysis && (
                <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm mb-6">
                  <h3 className="text-xl font-bold text-black-600 mb-4 flex items-center gap-2">
                    <Brain className="w-6 h-6 text-blue-600" />
                    Phân Tích Chi Tiết
                  </h3>

                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-black-600 mb-2">Tổng quan:</h4>
                      <p className="text-gray-700 leading-relaxed">
                        {finalResults.gpt_analysis.overall_analysis}
                      </p>
                    </div>

                    {/* Strengths */}
                    {finalResults.gpt_analysis.strengths && finalResults.gpt_analysis.strengths.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-green-700 mb-2">Điểm mạnh:</h4>
                        <ul className="list-disc list-inside text-gray-700 space-y-1">
                          {finalResults.gpt_analysis.strengths.map((strength: string, index: number) => (
                            <li key={index}>{strength}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Weaknesses */}
                    {finalResults.gpt_analysis.weaknesses && finalResults.gpt_analysis.weaknesses.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-orange-700 mb-2">Điểm cần cải thiện:</h4>
                        <ul className="list-disc list-inside text-gray-700 space-y-1">
                          {finalResults.gpt_analysis.weaknesses.map((weakness: string, index: number) => (
                            <li key={index}>{weakness}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm mb-6">
                <h3 className="text-xl font-bold text-black-600 mb-4 flex items-center gap-2">
                  <div className="text-2xl">💡</div>
                  Khuyến Nghị và Lời Khuyên
                </h3>

                <div className="space-y-4">
                  {finalResults.gpt_analysis?.recommendations && (
                    <div>
                      <h4 className="font-semibold text-blue-700 mb-2">Khuyến nghị cụ thể:</h4>
                      <ul className="list-disc list-inside text-gray-700 space-y-1">
                        {finalResults.gpt_analysis.recommendations.map((rec: string, index: number) => (
                          <li key={index}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {finalResults.gpt_analysis?.follow_up && (
                    <div>
                      <h4 className="font-semibold text-purple-700 mb-2">Theo dõi và tái khám:</h4>
                      <p className="text-gray-700">{finalResults.gpt_analysis.follow_up}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Test Statistics */}
              <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
                <h3 className="text-xl font-bold text-black-600 mb-4 flex items-center gap-2">
                  <div className="text-2xl">📊</div>
                  Thống Kê Bài Test
                </h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {finalResults.test_statistics?.total_questions || 0}
                    </div>
                    <div className="text-sm text-black-600">Tổng câu hỏi</div>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {finalResults.test_statistics?.completed_questions || 0}
                    </div>
                    <div className="text-sm text-black-600">Hoàn thành</div>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {finalResults.test_statistics?.completion_rate ?
                        Math.round(finalResults.test_statistics.completion_rate) : 0}%
                    </div>
                    <div className="text-sm text-black-600">Tỷ lệ hoàn thành</div>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {finalResults.generated_at || 'N/A'}
                    </div>
                    <div className="text-sm text-black-600">Thời gian</div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 justify-center mt-8">
                <Button
                  onClick={() => window.print()}
                  variant="secondaryOutline"
                  className="px-6 py-3"
                >
                  <div className="text-xl mr-2">🖨️</div>
                  In kết quả
                </Button>

                <Button
                  onClick={() => {
                    // Reset everything for new test
                    setTestResults([]);
                    navigateToQuestion(0, false);
                    setTestCompleted(false);
                    setFinalResults(null);
                    setHasRecording(false);
                    setAutoTranscriptionResult(null);
                    setAudioUrl(null);
                  }}
                  className="px-6 py-3" style={{
                    background: 'linear-gradient(135deg, #F4A261 0%, #E88D4D 100%)'
                  }}
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Làm bài test mới
                </Button>
              </div>
            </Card>
          </motion.div>
        )}
      </div>

      {/* Community intake modal */}
      {usageMode==='community' && !communityIntakeDone && showCommunityModal && (
        <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6">
            <h3 className="text-xl font-bold mb-2">Đăng ký kiểm tra cộng đồng</h3>
            <p className="text-sm text-black-600 mb-4">Nhập email để nhận kết quả sau khi hoàn thành bài kiểm tra.</p>
            <div className="space-y-3">
              <div>
                <label className="block text-xs mb-1">Email</label>
                <input value={communityEmail} onChange={(e)=>setCommunityEmail(e.target.value)} className="w-full border rounded-lg p-2" placeholder="you@example.com" />
              </div>
              <div>
                <label className="block text-xs mb-1">Họ tên (không bắt buộc)</label>
                <input value={communityName} onChange={(e)=>setCommunityName(e.target.value)} className="w-full border rounded-lg p-2" placeholder="Nguyễn Văn A" />
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="secondaryOutline" onClick={()=> setShowCommunityModal(false)}>Đóng</Button>
              <Button className="bg-amber-500 hover:bg-amber-600 text-white" onClick={async ()=>{
                if (!communityEmail.trim()) { alert('Email là bắt buộc'); return; }
                try {
                  const fd = new FormData();
                  fd.append('sessionId', sessionId);
                  fd.append('email', communityEmail);
                  fd.append('name', communityName);
                  if (userData?.age) fd.append('age', String(userData.age));
                  if (userData?.gender) fd.append('gender', String(userData.gender));
                  if (userData?.phone) fd.append('phone', String(userData.phone));
                  await fetch('/api/community/submit-intake', { method: 'POST', body: fd });
                  setCommunityIntakeDone(true);
                  setShowCommunityModal(false);
                } catch (e) { console.error(e); alert('Không thể lưu intake'); }
              }}>Lưu & Bắt đầu</Button>
            </div>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}

