// components/memory-test/TTSStatusIndicator.tsx
// Component hiển thị trạng thái TTS

import { CheckCircle, XCircle, Loader2, AlertTriangle, Volume2, VolumeX } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface TTSStatusIndicatorProps {
  status: {
    apiAvailable: boolean;
    webSpeechAvailable: boolean;
    vietnameseVoicesCount: number;
    error?: string;
  } | null;
  isLoading: boolean;
  isPlaying: boolean;
  error: string | null;
  progress: number;
}

export function TTSStatusIndicator({ 
  status, 
  isLoading, 
  isPlaying, 
  error, 
  progress 
}: TTSStatusIndicatorProps) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-600">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Kiểm tra hệ thống âm thanh...</span>
      </div>
    );
  }

  if (!status) return null;

  const getStatusColor = () => {
    if (error) return 'text-red-600';
    if (status.apiAvailable || status.webSpeechAvailable) return 'text-green-600';
    return 'text-orange-600';
  };

  const getStatusIcon = () => {
    if (error) return <XCircle className="w-4 h-4" />;
    if (status.apiAvailable || status.webSpeechAvailable) return <CheckCircle className="w-4 h-4" />;
    return <AlertTriangle className="w-4 h-4" />;
  };

  const getStatusText = () => {
    if (error) return 'Lỗi hệ thống âm thanh';
    if (status.apiAvailable) return 'TTS API hoạt động tốt';
    if (status.webSpeechAvailable) return 'Web Speech API khả dụng';
    return 'Không có hệ thống TTS';
  };

  return (
    <div className="space-y-2">
      <div className={`flex items-center gap-2 text-sm ${getStatusColor()}`}>
        {getStatusIcon()}
        <span>{getStatusText()}</span>
        {isPlaying && <Volume2 className="w-4 h-4 animate-pulse" />}
      </div>
      
      {status.vietnameseVoicesCount > 0 && (
        <div className="text-xs text-gray-600">
          {status.vietnameseVoicesCount} giọng tiếng Việt khả dụng
        </div>
      )}
      
      {isPlaying && progress > 0 && (
        <div className="w-full bg-gray-200 rounded-full h-1">
          <motion.div 
            className="bg-blue-500 h-1 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>
      )}
      
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="text-xs text-red-600 bg-red-50 p-2 rounded border border-red-200"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

