// hooks/useChatWithTTS.ts
'use client';

import { useState, useCallback, useRef } from 'react';
import { useGemini } from './useGemini';
import { ChatWithTTSRequest } from '../types/gemini';

export function useChatWithTTS() {
  const { sendMessage, chatHistory, isLoading: isChatLoading, error: chatError, clearHistory } = useGemini();
  const [isTTSLoading, setIsTTSLoading] = useState(false);
  const [ttsError, setTTSError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const generateTTS = useCallback(async (text: string, language = 'vi', speed = 'normal') => {
    setIsTTSLoading(true);
    setTTSError(null);

    try {
      const response = await fetch('/api/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          language,
          speed
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.details || errorData.error);
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      return audioUrl;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate speech';
      setTTSError(errorMessage);
      console.error('TTS Error:', err);
      return null;
    } finally {
      setIsTTSLoading(false);
    }
  }, []);

  const playAudio = useCallback((audioUrl: string) => {
    return new Promise<void>((resolve, reject) => {
      if (audioRef.current) {
        audioRef.current.pause();
        if (audioRef.current.src.startsWith('blob:')) {
          URL.revokeObjectURL(audioRef.current.src);
        }
      }

      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      
      audio.onloadstart = () => setIsPlaying(true);
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        resolve();
      };
      audio.onerror = (error) => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        reject(new Error('Failed to play audio'));
      };

      audio.play().catch(reject);
    });
  }, []);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      if (audioRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioRef.current.src);
      }
      setIsPlaying(false);
    }
  }, []);

  const sendMessageWithTTS = useCallback(async (request: ChatWithTTSRequest) => {
    // Clear previous TTS error
    setTTSError(null);
    
    // Send message to Gemini first
    const chatResponse = await sendMessage(request);
    
    if (!chatResponse) {
      return null;
    }

    // Generate TTS if enabled
    if (request.enableTTS && chatResponse.response) {
      const audioUrl = await generateTTS(
        chatResponse.response,
        request.ttsLanguage || 'vi',
        request.ttsSpeed || 'normal'
      );

      if (audioUrl) {
        // Auto-play the response
        try {
          await playAudio(audioUrl);
        } catch (err) {
          console.error('Auto-play failed:', err);
          setTTSError('Failed to play audio');
        }
      }
    }

    return {
      ...chatResponse,
      hasAudio: request.enableTTS || false
    };
  }, [sendMessage, generateTTS, playAudio]);

  const speakText = useCallback(async (text: string, language = 'vi', speed = 'normal') => {
    const audioUrl = await generateTTS(text, language, speed);
    if (audioUrl) {
      try {
        await playAudio(audioUrl);
      } catch (err) {
        console.error('Play audio failed:', err);
        setTTSError('Failed to play audio');
      }
    }
  }, [generateTTS, playAudio]);

  // Combined error handling
  const combinedError = chatError || ttsError;

  return {
    // Chat functions
    sendMessageWithTTS,
    chatHistory,
    clearHistory,
    
    // TTS functions
    generateTTS,
    playAudio,
    stopAudio,
    speakText,
    
    // State
    isChatLoading,
    isTTSLoading,
    isPlaying,
    chatError,
    ttsError,
    
    // Combined states
    isLoading: isChatLoading || isTTSLoading,
    error: combinedError
  };
}