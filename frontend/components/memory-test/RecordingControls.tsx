// components/memory-test/RecordingControls.tsx
// Component điều khiển ghi âm

import { Mic, Square, RotateCcw, CheckCircle, Timer } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';

interface RecordingControlsProps {
  isRecording: boolean;
  hasRecording: boolean;
  recordingDuration: number;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onResetRecording: () => void;
  disabled?: boolean;
}

export function RecordingControls({
  isRecording,
  hasRecording,
  recordingDuration,
  onStartRecording,
  onStopRecording,
  onResetRecording,
  disabled = false
}: RecordingControlsProps) {
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
            <span>Đã ghi âm xong!</span>
          </div>
          
          <div className="text-sm text-gray-600 text-center space-y-1">
            <div className="flex items-center justify-center gap-1">
              <Timer className="w-4 h-4" />
              <span>Thời lượng: {formatDuration(recordingDuration)}</span>
            </div>
            <div className="text-xs">File đã được lưu và đang xử lý...</div>
          </div>
        </div>
        
        <div className="flex justify-center">
          <Button
            onClick={onResetRecording}
            variant="primaryOutline"
            className="px-4 py-2 rounded-xl"
            disabled={disabled}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Ghi lại
          </Button>
        </div>
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
          onClick={onStopRecording}
          className="bg-red-600 hover:bg-red-700 text-white px-8 py-4 rounded-2xl text-lg font-bold shadow-lg"
          disabled={disabled}
        >
          <Square className="w-6 h-6 mr-3" />
          Dừng ghi âm
        </Button>
        
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <div className="flex items-center justify-center gap-3 text-red-600 font-medium mb-2">
            <motion.div 
              className="w-3 h-3 bg-red-500 rounded-full"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ repeat: Infinity, duration: 1 }}
            />
            <span>Đang ghi âm...</span>
            <Timer className="w-4 h-4" />
            <span className="font-mono text-lg">{formatDuration(recordingDuration)}</span>
          </div>
          
          <div className="text-sm text-gray-600 text-center">
            Hãy trả lời câu hỏi rõ ràng và nhấn &quot;Dừng ghi âm&quot; khi xong
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Button
        onClick={onStartRecording}
        className="bg-red-500 hover:bg-red-600 text-white px-8 py-4 rounded-2xl text-lg font-bold shadow-lg"
        disabled={disabled}
      >
        <Mic className="w-6 h-6 mr-3" />
        Bắt đầu ghi âm
      </Button>
      
      <div className="text-sm text-gray-500 mt-3">
        Nhấn để bắt đầu trả lời câu hỏi
      </div>
    </motion.div>
  );
}

