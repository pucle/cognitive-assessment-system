"use client";

import React, { useState, useRef, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Mic, MicOff, Square, Loader2 } from "lucide-react";

interface AudioRecorderProps {
  questionId: string;
  sessionId: string;
  onRecordingComplete: (data: {
    tempId: string;
    transcript: string;
    audioPath: string;
    confidence: number;
  }) => void;
  onError: (error: string) => void;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  questionId,
  sessionId,
  onRecordingComplete,
  onError
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);

  const timerRef = useRef<NodeJS.Timeout>();
  const streamRef = useRef<MediaStream>();

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      streamRef.current = stream;
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const chunks: Blob[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        await processAudio(audioBlob);
      };

      setRecordedChunks(chunks);
      setMediaRecorder(recorder);
      recorder.start(100); // Collect data every 100ms
      setIsRecording(true);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Recording failed:', error);
      onError('Không thể truy cập microphone. Vui lòng kiểm tra quyền truy cập.');
    }
  }, [onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);

      // Stop all tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  }, [mediaRecorder, isRecording]);

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, `recording_${questionId}.webm`);
      formData.append('questionId', questionId);
      formData.append('sessionId', sessionId);

      const response = await fetch('/api/audio/process', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.success) {
        onRecordingComplete({
          tempId: result.tempId,
          transcript: result.transcript,
          audioPath: result.audioPath,
          confidence: result.confidence
        });
      } else {
        onError(result.error || 'Xử lý audio thất bại');
      }

    } catch (error) {
      console.error('Audio processing error:', error);
      onError('Lỗi kết nối khi xử lý audio');
    } finally {
      setIsProcessing(false);
      setRecordingTime(0);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle className="text-center flex items-center justify-center gap-2">
          {isRecording ? (
            <Mic className="w-5 h-5 text-red-500 animate-pulse" />
          ) : (
            <MicOff className="w-5 h-5 text-gray-500" />
          )}
          {isRecording ? 'Đang ghi âm' : 'Sẵn sàng ghi âm'}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {isRecording && (
          <div className="text-center">
            <div className="text-2xl font-mono font-bold text-red-500">
              {formatTime(recordingTime)}
            </div>
            <div className="text-sm text-gray-600 mt-1">
              Nhấn dừng để hoàn thành
            </div>
          </div>
        )}

        {isProcessing && (
          <div className="text-center">
            <Loader2 className="w-8 h-8 animate-spin mx-auto text-blue-500" />
            <div className="text-sm text-gray-600 mt-2">
              Đang xử lý audio...
            </div>
          </div>
        )}

        <div className="flex gap-2">
          {!isRecording && !isProcessing && (
            <Button
              onClick={startRecording}
              className="flex-1"
              variant="default"
            >
              <Mic className="w-4 h-4 mr-2" />
              Bắt đầu ghi âm
            </Button>
          )}

          {isRecording && (
            <Button
              onClick={stopRecording}
              className="flex-1"
              variant="danger"
            >
              <Square className="w-4 h-4 mr-2" />
              Dừng ghi âm
            </Button>
          )}
        </div>

        <div className="text-xs text-gray-500 text-center">
          {!isRecording && !isProcessing && "Nhấn bắt đầu để ghi âm câu trả lời của bạn"}
          {isRecording && "Nói rõ ràng và tự nhiên"}
          {isProcessing && "Vui lòng đợi trong giây lát"}
        </div>
      </CardContent>
    </Card>
  );
};
