// components/memory-test/QuestionCard.tsx
// Component hiển thị câu hỏi

import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';

interface Question {
  id: number;
  category: string;
  text: string;
  instruction?: string;
}

interface QuestionCardProps {
  question: Question;
  greeting: string;
  questionIndex: number;
  totalQuestions: number;
  children?: React.ReactNode;
}

export function QuestionCard({ 
  question, 
  greeting, 
  questionIndex, 
  totalQuestions, 
  children 
}: QuestionCardProps) {
  return (
    <motion.div
      key={questionIndex}
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -50 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-4xl"
    >
      <Card className="p-8 backdrop-blur-xl bg-white/95 shadow-2xl rounded-3xl border border-white/30">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div className="inline-block bg-purple-100 text-purple-700 px-4 py-2 rounded-full text-sm font-medium">
            {question.category}
          </div>
          
          <div className="text-sm text-gray-500 font-medium">
            Câu {questionIndex + 1}/{totalQuestions}
          </div>
        </div>

        {/* Question Text */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 leading-relaxed">
            {question.text.replace('{greeting}', greeting)}
          </h2>
          
          {question.instruction && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-gray-600 bg-blue-50 p-4 rounded-xl border border-blue-200"
            >
              <div className="flex items-start gap-2">
                <span className="text-blue-600 font-medium">💡</span>
                <div>
                  <span className="font-medium text-blue-700">Hướng dẫn:</span>
                  <span className="ml-2">{question.instruction}</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Content from parent */}
        {children}
      </Card>
    </motion.div>
  );
}
