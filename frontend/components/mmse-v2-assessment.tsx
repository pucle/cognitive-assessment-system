"use client"

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ProgressBar } from '@/components/ui/progressbar'
import { LanguageSwitcher } from '@/components/LanguageSwitcher'
import { useLanguage } from '@/contexts/LanguageContext'
import { useRobustTTS } from '@/lib/tts-service'
import {
  Mic,
  MicOff,
  Upload,
  Loader2,
  Brain,
  Clock,
  User,
  FileAudio,
  CheckCircle,
  AlertCircle,
  XCircle,
  Info,
  Waves,
  Volume2,
  Pause,
  Square,
  Play
} from 'lucide-react'

interface MMSEQuestion {
  id: string
  domain: string
  question_text: string
  answer_type: string
  scoring_rule: string
  max_points: number
  sample_correct?: string | string[]
  sample_incorrect?: string | string[]
}

interface MMSEResult {
  session_id: string
  status: string
  timestamp: string
  mmse_scores: {
    M_raw: number
    L_scalar: number
    A_scalar: number
    final_score: number
    ml_prediction?: number
  }
  item_scores: Record<string, number>
  item_confidences: Record<string, number>
  features: {
    linguistic: { TTR: number; idea_density: number; F_flu: number; word_count: number }
    acoustic: { speech_rate_wpm: number; pause_rate: number; f0_variability: number; f0_mean: number }
  }
  transcription: { text: string; confidence: number; language: string }
  cognitive_status: {
    status: string
    risk_level: string
    description: string
    primary_score: number
    secondary_score?: number
    confidence: number
    recommendations: string[]
  }
  patient_info?: Record<string, any>
}

interface PatientInfo {
  name?: string
  age?: number
  gender?: string
  education_years?: number
  notes?: string
}

type UserData = { id?: string; name?: string; age?: number | string; gender?: string; email?: string; phone?: string }

export default function MMSEv2Assessment() {
  const { t, language } = useLanguage()

  // Track mounted to avoid hydration mismatches for client-only values
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  // Data state
  const [questions, setQuestions] = useState<MMSEQuestion[]>([])
  const [result, setResult] = useState<MMSEResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({})
  const [userData, setUserData] = useState<UserData | null>(null)
  const [greeting, setGreeting] = useState<string>('')

  // Question navigation
  const [currentIndex, setCurrentIndex] = useState(0)
  const currentQuestion = questions[currentIndex]

  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [currentAudio, setCurrentAudio] = useState<Blob | null>(null)
  const [isMicInitializing, setIsMicInitializing] = useState(false)
  const [isRecordingStarted, setIsRecordingStarted] = useState(false)
  const [backendStatus, setBackendStatus] = useState<{ ok: boolean; message: string } | null>(null)
  const [isBackendChecking, setIsBackendChecking] = useState<boolean>(false)

  // Generate sessionId on client only to avoid SSR mismatch
  const [sessionId, setSessionId] = useState<string>("")
  useEffect(() => {
    const sid = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
    setSessionId(sid)
  }, [])

  // TTS
  const { speak, pause, resume, stop, isSpeaking, isPaused, status: ttsStatus, error: ttsError } = useRobustTTS()

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const questionsAbortRef = useRef<AbortController | null>(null)
  const assessAbortRef = useRef<AbortController | null>(null)

  // Derived state
  const progress = questions.length > 0 ? ((currentIndex + (result ? 1 : 0)) / Math.max(questions.length, 1)) * 100 : 0

  // Derive friendly error for display
  const friendlyError = useMemo(() => {
    const raw = error || ttsError
    if (!raw) return null
    const msg = String(raw)
    if (msg.toLowerCase().includes('audio file not generated') && msg.toLowerCase().includes('enoent')) {
      return 'Chưa có file ghi âm nào, hãy tiến hành ngay nhé!'
    }
    return msg
  }, [error, ttsError])

  // Generate greeting from stored user data (similar to cognitive-assessment)
  const generateGreeting = useCallback((data: UserData | null, lang: string) => {
    if (!data || !data.name) { setGreeting('Chào mừng'); return }
    const nameParts = data.name.trim().split(/\s+/)
    let displayName = ''
    if (nameParts.length > 2) displayName = nameParts.slice(-2).join(' ')
    else if (nameParts.length === 2) displayName = nameParts.join(' ')
    else displayName = nameParts[0]

    const ageNum = typeof data.age === 'string' ? parseInt(data.age) : (data.age || 25)
    let honorific = ''
    if ((ageNum as number) >= 60) {
      honorific = (data.gender || 'Nam') === 'Nam' ? (lang === 'vi' ? 'ông' : 'Sir') : (lang === 'vi' ? 'bà' : 'Madam')
    } else if ((ageNum as number) >= 30) {
      honorific = (data.gender || 'Nam') === 'Nam' ? (lang === 'vi' ? 'anh' : 'Mr.') : (lang === 'vi' ? 'chị' : 'Ms.')
    } else {
      honorific = ''
    }
    setGreeting(`${honorific} ${displayName}`.trim())
  }, [])

  const checkBackend = useCallback(async () => {
    try {
      setIsBackendChecking(true)
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 2500)
      // Align with cognitive-assessment: use /api/health for a quick heartbeat
      const res = await fetch('http://localhost:5001/api/health', { signal: controller.signal })
      clearTimeout(timeout)
      if (res.ok) setBackendStatus({ ok: true, message: 'Online' })
      else setBackendStatus({ ok: false, message: 'Backend không phản hồi' })
    } catch {
      setBackendStatus({ ok: false, message: 'Không kết nối được backend' })
    } finally {
      setIsBackendChecking(false)
    }
  }, [])

  useEffect(() => {
    checkBackend()
    const interval = setInterval(checkBackend, 15000)
    return () => clearInterval(interval)
  }, [checkBackend])

  // Load questions with timeout and abort handling
  useEffect(() => {
    try {
      const stored = typeof window !== 'undefined' ? localStorage.getItem('userData') : null
      if (stored) {
        const parsed = JSON.parse(stored)
        setUserData(parsed)
        generateGreeting(parsed, language)
      }
    } catch {}
  }, [language, generateGreeting])

  // Style mapping for backend status pill (green/red/yellow + blinking)
  const backendStyles = useMemo(() => {
    if (isBackendChecking) return { text: 'text-amber-700', dot: 'bg-amber-500' }
    if (backendStatus?.ok) return { text: 'text-green-700', dot: 'bg-green-500' }
    return { text: 'text-rose-700', dot: 'bg-rose-500' }
  }, [isBackendChecking, backendStatus])

  // Load questions with timeout and abort handling
  useEffect(() => {
    const controller = new AbortController()
    questionsAbortRef.current = controller

    const timeoutId = setTimeout(() => controller.abort(), 10000)

    const loadQuestions = async () => {
      try {
        const res = await fetch('http://localhost:5001/api/mmse/questions', { signal: controller.signal })
        const data = await res.json().catch(() => ({ success: false }))
        if (res.ok && data.success) setQuestions(data.data.questions)
        else if (!controller.signal.aborted) setError('Không tải được danh sách câu hỏi')
      } catch (e: any) {
        if (e?.name === 'AbortError') {
          // Silent on abort
        } else {
          setError('Lỗi khi tải câu hỏi MMSE')
        }
      } finally {
        clearTimeout(timeoutId)
      }
    }

    loadQuestions()

    return () => {
      clearTimeout(timeoutId)
      controller.abort()
    }
  }, [])

  // Timer for recording duration – only start when MediaRecorder actually starts
  useEffect(() => {
    if (!isRecordingStarted) return
    timerRef.current = setInterval(() => setRecordingDuration((d) => d + 1), 1000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [isRecordingStarted])

  // Recording
  const startRecording = useCallback(async () => {
    try {
      setIsMicInitializing(true)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm'
      })
      audioChunksRef.current = []
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data) }
      mediaRecorder.onstart = () => {
        setIsMicInitializing(false)
        setIsRecordingStarted(true)
      }
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        setCurrentAudio(blob)
        stream.getTracks().forEach((t) => t.stop())
        setIsRecordingStarted(false)
      }
      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
      setRecordingDuration(0)
      setError(null)
    } catch (e) {
      setIsMicInitializing(false)
      setIsRecording(false)
      setError('Không thể bắt đầu ghi âm')
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) clearInterval(timerRef.current)
      setIsMicInitializing(false)
      setIsRecordingStarted(false)
    }
  }, [isRecording])

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setCurrentAudio(file)
      setError(null)
    }
  }, [])

  const nextQuestion = () => setCurrentIndex((i) => Math.min(i + 1, Math.max(questions.length - 1, 0)))
  const prevQuestion = () => setCurrentIndex((i) => Math.max(i - 1, 0))

  // Map code prefix to meaning & role in Vietnamese
  const getCodeInfo = useCallback((id: string): { title: string; role: string } => {
    const match = id?.toLowerCase().match(/^([a-z]+)(\d+)/)
    const prefix = match?.[1] || id?.[0]?.toLowerCase() || ''
    switch (prefix) {
      case 't':
        return { title: 'Định hướng thời gian', role: 'Đánh giá nhận thức về ngày, tháng, năm, thứ' }
      case 'p':
        return { title: 'Định hướng địa điểm', role: 'Đánh giá nhận thức về nơi chốn, vị trí hiện tại' }
      case 'r':
        return { title: 'Ghi nhớ', role: 'Đánh giá khả năng ghi nhớ ngắn hạn ngay lập tức' }
      case 'a':
        return { title: 'Chú ý/Tính toán', role: 'Đánh giá sự tập trung và thao tác tính đơn giản' }
      case 'm':
        return { title: 'Nhớ lại trì hoãn', role: 'Đánh giá khả năng gợi nhớ sau khoảng thời gian' }
      case 'l':
        return { title: 'Ngôn ngữ', role: 'Đánh giá gọi tên, lặp lại, làm theo mệnh lệnh' }
      default:
        return { title: 'Đánh giá chức năng', role: 'Đánh giá chức năng nhận thức liên quan' }
    }
  }, [])

  // Select question and auto-read via TTS in background
  const selectQuestion = useCallback(async (idx: number) => {
    setCurrentIndex(idx)
    const q = questions[idx]
    if (q?.question_text) {
      try {
        stop()
        const text = `${q.id}. ${(q.question_text || '').replace('{greeting}', greeting)}`
        await speak(text, { language: 'vi', rate: 0.95 })
      } catch {}
    }
  }, [questions, speak, stop, greeting])

  const playQuestionTTS = async () => {
    if (!currentQuestion) return
    const text = `${currentQuestion.id}. ${(currentQuestion.question_text || '').replace('{greeting}', greeting)}`
    try {
      await speak(text, { language: 'vi', rate: 0.95 })
    } catch {}
  }

  // Process assessment
  const processAssessment = async () => {
    if (!currentAudio) { setError('Chưa có file audio để đánh giá'); return }
    setIsProcessing(true)
    setError(null)
    setResult(null)

    // Abort any previous assessment request
    if (assessAbortRef.current) assessAbortRef.current.abort()
    const controller = new AbortController()
    assessAbortRef.current = controller

    const timeoutId = setTimeout(() => controller.abort(), 120000) // 120s timeout cho MMSE

    try {
      const formData = new FormData()
      formData.append('audio', currentAudio, 'mmse_session.webm')
      if (sessionId) formData.append('session_id', sessionId)
      if (Object.keys(patientInfo).length > 0) formData.append('patient_info', JSON.stringify(patientInfo))

      const response = await fetch('http://localhost:5001/api/mmse/assess', { method: 'POST', body: formData, signal: controller.signal })
      const data = await response.json().catch(() => ({ success: false }))

      if (response.ok && data.success) {
        setResult(data.data)
        // Đợi một chút để đảm bảo UI render xong, sau đó mới tắt trạng thái processing
        setTimeout(() => {
          setIsProcessing(false)
        }, 500) // 500ms delay để UI có thời gian render kết quả
      } else if (!controller.signal.aborted) {
        setError(data.error || 'Đánh giá thất bại')
        setIsProcessing(false) // Chỉ tắt khi có lỗi
      }
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        // Likely timeout or component unmounted – do not surface noisy error
        setIsProcessing(false)
      } else {
        setError('Lỗi khi gửi đánh giá đến server')
        setIsProcessing(false)
      }
    } finally {
      clearTimeout(timeoutId)
    }
  }

  const getRiskBadge = (risk: string) => {
    const map: Record<string, string> = {
      low: 'bg-green-100 text-green-800',
      moderate: 'bg-yellow-100 text-yellow-800',
      high: 'bg-orange-100 text-orange-800',
      very_high: 'bg-red-100 text-red-800'
    }
    return map[risk] || 'bg-gray-100 text-gray-800'
  }

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (questionsAbortRef.current) questionsAbortRef.current.abort()
      if (assessAbortRef.current) assessAbortRef.current.abort()
    }
  }, [])

  return (
    <div className="max-w-7xl mx-auto p-4 sm:p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Brain className="w-8 h-8 text-blue-600" />
          <h1 className="text-2xl sm:text-3xl font-extrabold text-blue-700">Đánh giá MMSE v2</h1>
        </div>
        <div className="flex items-center gap-3">
          {/* Backend status pill with animation, aligned to cognitive-assessment style */}
          {backendStatus && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`inline-flex items-center gap-2`}
            >
              <span className={`w-3 h-3 rounded-full ${backendStyles.dot} ${isBackendChecking ? 'animate-ping' : ''}`} />
              <span className={`text-sm font-bold ${backendStyles.text}`}>
                {isBackendChecking ? 'Checking…' : (backendStatus.ok ? 'Connected' : 'Disconnected')}
              </span>
            </motion.div>
          )}
          <LanguageSwitcher />
        </div>
      </div>

      {friendlyError && (
        <Alert variant="destructive">
          <AlertDescription>{friendlyError}</AlertDescription>
        </Alert>
      )}

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Questions & Recording */}
        <div className="lg:col-span-2 space-y-4">
          <Card variant="underwater">
            <CardHeader>
              <CardTitle variant="underwater" className="flex items-center gap-2">
                <Waves className="w-5 h-5" /> Câu hỏi MMSE
              </CardTitle>
              <CardDescription>
                Chọn câu hỏi và nhấn loa để hệ thống đọc, sau đó ghi âm câu trả lời.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Progress */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>Tiến độ</span>
                  <span>{questions.length > 0 ? `${currentIndex + 1}/${questions.length}` : '0/0'}</span>
                </div>
                <ProgressBar value={progress} />
              </div>

              {/* Question list (only symbol + meaning/role) */}
              <div className="rounded-xl border bg-white/70 divide-y max-h-64 overflow-auto">
                {questions.map((q, idx) => (
                  <div key={q.id} className={`p-3 cursor-pointer ${idx === currentIndex ? 'bg-blue-50' : 'hover:bg-gray-50'}`} onClick={() => selectQuestion(idx)}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge>{q.id}</Badge>
                        <span className="font-medium">{getCodeInfo(q.id).title}</span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 mt-1">Vai trò: {getCodeInfo(q.id).role}</div>
                  </div>
                ))}
              </div>

              {/* TTS controls */}
              {currentQuestion && (
                <div className="flex items-center gap-2">
                  <Button variant="primaryOutline" onClick={playQuestionTTS}><Volume2 className="w-4 h-4 mr-2" /> Đọc câu hỏi</Button>
                  {!isPaused && isSpeaking && <Button variant="primaryOutline" onClick={pause}><Pause className="w-4 h-4 mr-2" /> Tạm dừng</Button>}
                  {isPaused && <Button variant="primaryOutline" onClick={resume}><Play className="w-4 h-4 mr-2" /> Tiếp tục</Button>}
                  <Button variant="primaryOutline" onClick={stop}><Square className="w-4 h-4 mr-2" /> Dừng</Button>
                </div>
              )}

              {/* Recording controls */}
              <div className="grid sm:grid-cols-2 gap-3">
                <Button onClick={isRecording ? stopRecording : startRecording} variant={isRecording ? 'danger' : 'primary'} className="w-full">
                  {isRecording ? (
                    <>
                      <MicOff className="w-5 h-5 mr-2" /> Dừng ghi âm ({recordingDuration}s)
                    </>
                  ) : isMicInitializing ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" /> Đang khởi tạo mic…
                    </>
                  ) : (
                    <>
                      <Mic className="w-5 h-5 mr-2" /> Bắt đầu ghi âm
                    </>
                  )}
                </Button>
                <div>
                  <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept="audio/*" className="hidden" />
                  <Button variant="primaryOutline" className="w-full" onClick={() => fileInputRef.current?.click()}>
                    <FileAudio className="w-5 h-5 mr-2" /> Chọn file audio
                  </Button>
                </div>
              </div>

              {currentAudio && (
                <div className="flex items-center justify-between rounded-xl border bg-white/60 p-3 text-sm">
                  <div className="flex items-center gap-2 text-green-700">
                    <CheckCircle className="w-4 h-4" /> Audio đã sẵn sàng để xử lý
                  </div>
                  <Button onClick={processAssessment} disabled={isProcessing || !!result}>
                    {result ? 'Đã hoàn thành đánh giá' :
                     isProcessing ? (<><Clock className="w-4 h-4 mr-2 animate-spin" /> Đang tiến hành chấm điểm...</>) :
                     'Chạy đánh giá MMSE'}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          {(isProcessing || result) && (
            <Card variant="floating">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {result ? (
                    <>
                      {result.cognitive_status.status === 'normal' && <CheckCircle className="w-5 h-5 text-green-600" />}
                      {result.cognitive_status.status === 'mild_impairment' && <Info className="w-5 h-5 text-yellow-600" />}
                      {result.cognitive_status.status === 'moderate_impairment' && <AlertCircle className="w-5 h-5 text-orange-600" />}
                      {result.cognitive_status.status === 'severe_impairment' && <XCircle className="w-5 h-5 text-red-600" />}
                      Kết quả đánh giá
                    </>
                  ) : (
                    <>
                      <Clock className="w-5 h-5 text-blue-600 animate-spin" /> Đang xử lý...
                    </>
                  )}
                </CardTitle>
                {result && (
                  <CardDescription>
                    {mounted ? (
                      <>Phiên đánh giá: {result.session_id} • {new Date(result.timestamp).toLocaleString('vi-VN')}</>
                    ) : (
                      <>Phiên đánh giá: {result.session_id}</>
                    )}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-4">
                {!result && (
                  <div className="p-3 rounded-lg bg-blue-50 text-center text-sm text-blue-700">
                    Vui lòng chờ trong giây lát. Hệ thống đang chấm điểm và tổng hợp kết quả.
                  </div>
                )}

                {result && (
                  <>
                    <div className="grid sm:grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="text-3xl font-extrabold text-blue-600">{result.mmse_scores.final_score}/30</div>
                        <div className="text-sm text-gray-600">Điểm MMSE cuối</div>
                      </div>
                      <div className="text-center">
                        <Badge className={getRiskBadge(result.cognitive_status.risk_level)}>
                          {result.cognitive_status.risk_level.replace('_', ' ').toUpperCase()}
                        </Badge>
                        <div className="text-sm text-gray-600 mt-1">Mức độ nguy cơ</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold">{Math.round(result.cognitive_status.confidence * 100)}%</div>
                        <div className="text-sm text-gray-600">Độ tin cậy</div>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-gray-50 text-center">
                      <p className="text-sm font-medium">{result.cognitive_status.description}</p>
                    </div>

                    <Separator />

                    <div className="grid sm:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold mb-2">Linguistic</h4>
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between"><span>TTR</span><span>{result.features.linguistic.TTR.toFixed(3)}</span></div>
                          <div className="flex justify-between"><span>Idea density</span><span>{result.features.linguistic.idea_density.toFixed(3)}</span></div>
                          <div className="flex justify-between"><span>Fluency</span><span>{result.features.linguistic.F_flu.toFixed(3)}</span></div>
                          <div className="flex justify-between"><span>Word count</span><span>{result.features.linguistic.word_count}</span></div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">Acoustic</h4>
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between"><span>Speech rate</span><span>{Number(result.features.acoustic.speech_rate_wpm).toFixed(2)} WPM</span></div>
                          <div className="flex justify-between"><span>Pause rate</span><span>{(result.features.acoustic.pause_rate * 100).toFixed(2)}%</span></div>
                          <div className="flex justify-between"><span>F0 var</span><span>{Number(result.features.acoustic.f0_variability).toFixed(2)} Hz</span></div>
                          <div className="flex justify-between"><span>F0 mean</span><span>{Number(result.features.acoustic.f0_mean).toFixed(2)} Hz</span></div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Transcript</h4>
                      <div className="p-3 bg-gray-50 rounded-lg text-sm">
                        {result.transcription.text || 'Không có transcript'}
                      </div>
                    </div>

                    {result.cognitive_status.recommendations?.length > 0 && (
                      <div>
                        <h4 className="font-semibold mb-2">Khuyến nghị</h4>
                        <div className="space-y-2">
                          {result.cognitive_status.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-start gap-2 p-2 bg-blue-50 rounded">
                              <div className="w-2 h-2 bg-blue-600 rounded-full mt-2" />
                              <span className="text-sm">{rec}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Patient info & Status */}
        <div className="space-y-4">
          <Card variant="underwater">
            <CardHeader>
              <CardTitle variant="underwater" className="flex items-center gap-2"><User className="w-5 h-5" /> Thông tin người dùng</CardTitle>
              <CardDescription>Không bắt buộc nhưng giúp tăng độ chính xác</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Họ tên</label>
                  <input className="w-full p-2 rounded-lg border" value={patientInfo.name || ''} onChange={(e) => setPatientInfo((p) => ({ ...p, name: e.target.value }))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Tuổi</label>
                  <input type="number" min={0} max={120} className="w-full p-2 rounded-lg border" value={patientInfo.age || ''} onChange={(e) => setPatientInfo((p) => ({ ...p, age: parseInt(e.target.value) || undefined }))} />
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Giới tính</label>
                  <select className="w-full p-2 rounded-lg border" value={patientInfo.gender || ''} onChange={(e) => setPatientInfo((p) => ({ ...p, gender: e.target.value }))}>
                    <option value="">Chọn</option>
                    <option value="male">Nam</option>
                    <option value="female">Nữ</option>
                    <option value="other">Khác</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Số năm học (Trình độ học vấn)</label>
                  <input type="number" min={0} max={30} className="w-full p-2 rounded-lg border" value={patientInfo.education_years || ''} onChange={(e) => setPatientInfo((p) => ({ ...p, education_years: parseInt(e.target.value) || undefined }))} />
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">Ghi chú</label>
                <textarea rows={3} className="w-full p-2 rounded-lg border" value={patientInfo.notes || ''} onChange={(e) => setPatientInfo((p) => ({ ...p, notes: e.target.value }))} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Clock className="w-5 h-5" /> Trạng thái</CardTitle>
              <CardDescription>Phiên: {sessionId}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span>Đọc câu hỏi</span>
                <Badge className={isSpeaking ? 'bg-blue-50 text-blue-700' : 'bg-gray-50 text-gray-700'}>
                  {isPaused ? 'Tạm dừng' : (isSpeaking ? 'Đang đọc' : 'Tắt')}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>Ghi âm</span>
                <Badge className={isRecording ? 'bg-red-50 text-red-700' : 'bg-gray-50 text-gray-700'}>
                  {isRecording ? 'Đang ghi...' : 'Tạm dừng'}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>Thời lượng</span>
                <span className="font-medium">{recordingDuration}s</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Audio</span>
                <Badge className={currentAudio ? 'bg-green-50 text-green-700' : 'bg-gray-50 text-gray-700'}>
                  {currentAudio ? 'Đã chọn' : 'Chưa có'}
                </Badge>
              </div>
              {result && (
                <div className="flex items-center justify-between">
                  <span>Kết quả</span>
                  <span className="font-semibold text-blue-700">{result.mmse_scores.final_score}/30</span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
