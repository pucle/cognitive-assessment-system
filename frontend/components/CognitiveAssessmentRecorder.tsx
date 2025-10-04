"use client";

import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Mic, Square, RotateCcw, CheckCircle, Timer, Brain } from 'lucide-react';
import { motion } from 'framer-motion';
import { useLanguage } from '@/contexts/LanguageContext';

interface AssessmentResult {
  success: boolean;
  transcript?: string;
  analysis?: string;
  score?: number;
  error?: string;
}

interface CognitiveAssessmentRecorderProps {
  onAssessmentComplete: (result: AssessmentResult) => void;
  question?: string;
  disabled?: boolean;
}

export function CognitiveAssessmentRecorder({
  onAssessmentComplete,
  question = "Hãy mô tả một ngày trong tuần của bạn",
  disabled = false
}: CognitiveAssessmentRecorderProps) {
  const { t } = useLanguage();
  const [isRecording, setIsRecording] = useState(false);
  const [isRecordingStarted, setIsRecordingStarted] = useState(false);
  const [hasRecording, setIsHasRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [manualTranscript, setManualTranscript] = useState('');
  const [showTranscriptInput, setShowTranscriptInput] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100,
        }
      });
      
      streamRef.current = stream;
      chunksRef.current = [];
      
      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
          ? 'audio/webm;codecs=opus'
          : MediaRecorder.isTypeSupported('audio/webm') 
          ? 'audio/webm' 
          : 'audio/mp4'
      });
      
      recorder.onstart = () => {
        console.log('🎙️ Recording actually started');
        setIsRecordingStarted(true);

        // Start timer only when recording actually begins
        timerRef.current = setInterval(() => {
          setRecordingDuration(prev => prev + 1);
        }, 1000);
      };

      recorder.ondataavailable = (e) => {
        console.log('📊 Data available:', e.data.size, 'bytes');
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
      
      recorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { 
          type: recorder.mimeType || 'audio/webm' 
        });
        
        setIsRecording(false);
        setIsHasRecording(true);
        
        // Clean up stream
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        // Stop timer
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        
        // Store audio blob for later use
        (window as Window & { lastAudioBlob?: Blob }).lastAudioBlob = audioBlob;
      };

      recorder.onerror = (e) => {
        console.error('❌ MediaRecorder error:', e);
        cleanup();
      };

      mediaRecorderRef.current = recorder;
      recorder.start(1000);
      setIsRecording(true);
      setIsRecordingStarted(false);
      setIsHasRecording(false);
      setRecordingDuration(0);
      
    } catch (error) {
      console.error('❌ Error accessing microphone:', error);
      alert('Không thể truy cập microphone. Vui lòng kiểm tra quyền truy cập.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  };

  const resetRecording = () => {
    cleanup();
    setIsHasRecording(false);
    setIsRecordingStarted(false);
    setRecordingDuration(0);
    setManualTranscript('');
    setShowTranscriptInput(false);
  };

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
    setIsRecordingStarted(false);
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const submitAssessment = async () => {
    if (!manualTranscript.trim()) {
      alert('Vui lòng nhập transcript hoặc ghi âm');
      return;
    }

    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      
      // Add audio file if available
      if ((window as Window & { lastAudioBlob?: Blob }).lastAudioBlob) {
        formData.append('audio', (window as Window & { lastAudioBlob?: Blob }).lastAudioBlob!, 'recording.webm');
      }
      
      // Add form data
      formData.append('transcript', manualTranscript);
      formData.append('question', question);
      formData.append('user_id', 'user_' + Date.now());
      formData.append('age', '65'); // Default age
      formData.append('gender', 'unknown'); // Default gender
      
      const response = await fetch('http://localhost:5001/assess-cognitive', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ Assessment result:', result);
        onAssessmentComplete(result);
      } else {
        const errorData = await response.json();
        console.error('❌ Assessment failed:', errorData);
        alert(`Đánh giá thất bại: ${errorData.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('❌ Error submitting assessment:', error);
      alert('Lỗi kết nối đến server. Vui lòng kiểm tra backend.');
    } finally {
      setIsProcessing(false);
    }
  };

  if (hasRecording) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4"
      >
        <div className="bg-green-50 border border-green-200 rounded-xl p-4">
          <div className="flex items-center justify-center gap-2 text-green-700 font-medium mb-2">
            <CheckCircle className="w-5 h-5" />
            <span>{t('recording_done')}</span>
          </div>
          
          <div className="text-sm text-gray-600 text-center space-y-1">
            <div className="flex items-center justify-center gap-1">
              <Timer className="w-4 h-4" />
              <span>{t('duration')}: {formatDuration(recordingDuration)}</span>
            </div>
          </div>
        </div>

        {!showTranscriptInput ? (
          <div className="space-y-3">
            <Button
              onClick={() => setShowTranscriptInput(true)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl"
            >
              <Brain className="w-5 h-5 mr-2" />
              Nhập Transcript & Đánh giá
            </Button>
            
            <Button
              onClick={resetRecording}
              variant="secondaryOutline"
              className="w-full px-6 py-3 rounded-xl"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Ghi lại
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('transcript_label')}
              </label>
              <textarea
                value={manualTranscript}
                onChange={(e) => setManualTranscript(e.target.value)}
                placeholder={t('transcript_placeholder')}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={4}
              />
            </div>
            
            <div className="flex gap-3">
              <Button
                onClick={submitAssessment}
                disabled={isProcessing || !manualTranscript.trim()}
                className="flex-1 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-xl"
              >
                {isProcessing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    {t('processing')}
                  </>
                ) : (
                  <>
                    <Brain className="w-5 h-5 mr-2" />
                    {t('cognitive_assessment')}
                  </>
                )}
              </Button>
              
              <Button
                onClick={() => setShowTranscriptInput(false)}
                variant="secondaryOutline"
                className="px-6 py-3 rounded-xl"
              >
                {t('cancel')}
              </Button>
            </div>
          </div>
        )}
      </motion.div>
    );
  }

  if (isRecording) {
    return (
      <motion.div
        animate={{ scale: [1, 1.02, 1] }}
        transition={{ repeat: Infinity, duration: 1.5 }}
        className="space-y-4"
      >
        <Button
          onClick={stopRecording}
          className="bg-red-600 hover:bg-red-700 text-white px-8 py-4 rounded-2xl text-lg font-bold shadow-lg w-full"
          disabled={disabled}
        >
          <Square className="w-6 h-6 mr-3" />
          {t('stop_recording')}
        </Button>
        
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <div className="flex items-center justify-center gap-3 text-red-600 font-medium mb-2">
            <motion.div
              className="w-3 h-3 bg-red-500 rounded-full"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ repeat: Infinity, duration: 1 }}
            />
            <span>{isRecordingStarted ? 'Đang ghi âm...' : 'Đang chuẩn bị ghi âm...'}</span>
            {isRecordingStarted && (
              <>
                <Timer className="w-4 h-4" />
                <span className="font-mono text-lg">{formatDuration(recordingDuration)}</span>
              </>
            )}
          </div>

          <div className="text-sm text-gray-600 text-center">
            {isRecordingStarted
              ? t('answer_instruction')
              : t('initializing_microphone')
            }
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="space-y-4"
    >
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-4">
        <div className="flex items-center gap-2 text-blue-700 font-medium mb-2">
          <Brain className="w-5 h-5" />
          <span>{t('assessment_question')}</span>
        </div>
        <p className="text-gray-700 text-center">{question}</p>
      </div>
      
      <Button
        onClick={startRecording}
        className="bg-red-500 hover:bg-red-600 text-white px-8 py-4 rounded-2xl text-lg font-bold shadow-lg w-full"
        disabled={disabled}
      >
        <Mic className="w-6 h-6 mr-3" />
        {t('start_recording_instruction')}
      </Button>
      
      <div className="text-sm text-gray-500 text-center">
        {t('tap_to_start')}
      </div>
    </motion.div>
  );
}
