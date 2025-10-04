"use client";

import { CheckCircle, FileText, AlertCircle } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

interface TranscriptionStatusProps {
  textRecord: {
    saved: boolean;
    filename?: string;
    directory?: string;
  };
  method?: string;
  cost?: number;
  className?: string;
}

export default function TranscriptionStatus({
  textRecord,
  method,
  cost,
  className = ""
}: TranscriptionStatusProps) {
  const { t } = useLanguage();

  if (!textRecord) return null;

  return (
    <div className={`bg-gray-50 border border-gray-200 rounded-lg p-3 ${className}`}>
      <div className="flex items-center gap-2 mb-2">
        {textRecord.saved ? (
          <CheckCircle className="w-4 h-4 text-green-600" />
        ) : (
          <AlertCircle className="w-4 h-4 text-yellow-600" />
        )}
        <span className="font-medium text-sm text-gray-700">
          {t('transcription_status')}
        </span>
      </div>

      <div className="space-y-1 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <FileText className="w-3 h-3" />
          <span>
            {textRecord.saved ? t('saved') : t('not_saved')}: {textRecord.filename || t('no_filename')}
          </span>
        </div>

        {textRecord.directory && (
          <div>{t('directory')}: {textRecord.directory}</div>
        )}

        {method && (
          <div>{t('method')}: {method}</div>
        )}

        {cost !== undefined && (
          <div>{t('cost')}: ${cost.toFixed(4)}</div>
        )}
      </div>
    </div>
  );
}
