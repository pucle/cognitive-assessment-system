// types/gemini.ts
export type GeminiModel = 'gemini-pro' | 'gemini-pro-vision' | 'gemini-1.5-pro' | 'gemini-1.5-flash';

export interface GeminiChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
}

export interface GeminiChatRequest {
  message: string;
  model?: GeminiModel;
  systemPrompt?: string;
  chatHistory?: GeminiChatMessage[];
  temperature?: number;
  maxTokens?: number;
}

export interface GeminiChatResponse {
  success: boolean;
  response: string;
  model: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  timestamp: string;
}

export interface GeminiErrorResponse {
  error: string;
  details: string;
  timestamp: string;
}

export interface TTSRequest {
  text: string;
  language?: string;
  speed?: 'normal' | 'slow';
}

// Combined interface for chat with TTS
export interface ChatWithTTSRequest extends GeminiChatRequest {
  enableTTS?: boolean;
  ttsLanguage?: string;
  ttsSpeed?: 'normal' | 'slow';
}
// types/gemini.ts

export interface GeminiMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}
