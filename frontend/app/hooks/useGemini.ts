// hooks/useGemini.ts
'use client';

import { useState, useCallback } from 'react';
import { GeminiChatRequest, GeminiChatResponse, GeminiErrorResponse, GeminiChatMessage } from '../types/gemini';

export function useGemini() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<GeminiChatMessage[]>([]);

  const sendMessage = useCallback(async (
    request: GeminiChatRequest
  ): Promise<GeminiChatResponse | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...request,
          chatHistory: request.chatHistory || chatHistory
        }),
      });

      if (!response.ok) {
        const errorData: GeminiErrorResponse = await response.json();
        throw new Error(errorData.details || errorData.error);
      }

      const data: GeminiChatResponse = await response.json();
      
      // Update chat history
      const newMessages: GeminiChatMessage[] = [
        { role: 'user', content: request.message, timestamp: new Date().toISOString() },
        { role: 'assistant', content: data.response, timestamp: data.timestamp }
      ];
      
      setChatHistory(prev => [...prev, ...newMessages]);
      
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      console.error('Gemini API Error:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [chatHistory]);

  const clearHistory = useCallback(() => {
    setChatHistory([]);
    setError(null);
  }, []);

  const testConnection = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch('/api/chat?test=true');
      const data = await response.json();
      return data.status === 'success';
    } catch (err) {
      console.error('Connection test failed:', err);
      return false;
    }
  }, []);

  return {
    sendMessage,
    clearHistory,
    testConnection,
    chatHistory,
    isLoading,
    error,
  };
}