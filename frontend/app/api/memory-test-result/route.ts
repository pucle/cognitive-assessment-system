// app/api/memory-test-results/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { existsSync } from 'fs';

interface TestResult {
  questionId: number;
  question: string;
  audioBlob?: Blob;
  transcription?: string;
  timestamp: Date;
}

interface TestSubmission {
  userId: string;
  sessionId: string;
  results: TestResult[];
  completedAt: Date;
  userInfo?: {
    name?: string;
    age?: string;
    email?: string;
    phone?: string;
  };
}

interface CognitiveAssessmentResult {
  combined_assessment: {
    combined_score: number;
    max_score: number;
    audio_score: number;
    text_score: number;
    risk_level: string;
    recommendations: string[];
  };
  audio_features: Record<string, unknown>;
  text_analysis: Record<string, unknown>;
  participant_info: {
    age: number;
    gender: 'male' | 'female' | 'other';
  };
  timestamp: string;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Lấy dữ liệu từ FormData
    const userId = formData.get('userId') as string;
    const sessionId = formData.get('sessionId') as string;
    const resultsJson = formData.get('results') as string;
    const completedAt = formData.get('completedAt') as string;
    const userInfoJson = formData.get('userInfo') as string;

    if (!userId || !resultsJson) {
      return NextResponse.json({ error: 'Missing required data' }, { status: 400 });
    }

    const results: TestResult[] = JSON.parse(resultsJson);
    const userInfo = userInfoJson ? JSON.parse(userInfoJson) : {};

    // Đường dẫn cố định bạn muốn lưu
    const baseDir = join(process.cwd(), 'frontend', 'recordings');
    // Không tạo subfolder theo user nữa
    let userDir = baseDir;
    
    // Tạo thư mục nếu chưa tồn tại
    try {
      if (!existsSync(userDir)) {
        await mkdir(userDir, { recursive: true });
        console.log('Created directory:', userDir);
      }
    } catch (error) {
      console.error('Error creating directory:', error);
      // Fallback cũng về recordings
      const fallbackDir = join(process.cwd(), 'frontend', 'recordings');
      if (!existsSync(fallbackDir)) {
        await mkdir(fallbackDir, { recursive: true });
      }
      userDir = fallbackDir;
    }

    // Tạo ID cho test session
    const testId = sessionId || `test-${typeof window !== 'undefined' ? Date.now() : Math.floor(Math.random() * 1000000)}`;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Lưu file audio và phân tích cognitive assessment
    const audioFiles: string[] = [];
    const cognitiveResults: CognitiveAssessmentResult[] = [];
    
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const audioFile = formData.get(`audioFile_${i}`) as File;
      
      if (audioFile) {
        try {
          // Lưu file audio
          const audioFilename = `${testId}_q${result.questionId}_${timestamp}.webm`;
          const audioPath = join(userDir, audioFilename);
          
          const arrayBuffer = await audioFile.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          await writeFile(audioPath, buffer);
          
          audioFiles.push(audioFilename);
          console.log('Audio file saved:', audioPath);

          // Gửi file audio cho backend Python để phân tích (gọi luôn, có hoặc không có transcription)
          try {
            const cognitiveResult = await analyzeCognitiveAssessment(
              audioFile, 
              {
                age: parseInt(userInfo.age) || 30,
                gender: (userInfo.gender as 'male' | 'female' | 'other') || 'other'
              },
              result.transcription || '',
              userId,
              result.question || '',
              result.questionId
            );
            
            if (cognitiveResult) {
              cognitiveResults.push(cognitiveResult);
            }
          } catch (cognitiveError) {
            console.error(`Error analyzing cognitive assessment for question ${result.questionId}:`, cognitiveError);
          }
          
        } catch (audioError) {
          console.error(`Error saving audio for question ${result.questionId}:`, audioError);
        }
      }
    }

    // Tính điểm memory test
    const memoryScore = calculateMemoryScore(results);
    
    // Tính điểm cognitive assessment trung bình
    const avgCognitiveScore = cognitiveResults.length > 0 
      ? cognitiveResults.reduce((sum, result) => sum + result.combined_assessment.combined_score, 0) / cognitiveResults.length
      : 0;

    const resultData = {
      testId,
      userId,
      sessionId,
      userInfo,
      results: results.map(result => ({
        questionId: result.questionId,
        question: result.question,
        transcription: result.transcription,
        timestamp: result.timestamp,
        hasAudio: !!formData.get(`audioFile_${results.indexOf(result)}`)
      })),
      completedAt,
      totalQuestions: results.length,
      answeredQuestions: results.filter(r => r.transcription).length,
      memoryScore,
      cognitiveAssessment: {
        averageScore: avgCognitiveScore,
        results: cognitiveResults,
        totalAnalyzed: cognitiveResults.length
      },
      createdAt: new Date(),
      savedLocation: userDir,
      audioFiles
    };

    // Lưu file JSON với thông tin chi tiết
    const resultsFilename = `${testId}_results_${timestamp}.json`;
    const resultsPath = join(userDir, resultsFilename);
    
    try {
      await writeFile(resultsPath, JSON.stringify(resultData, null, 2), 'utf8');
      console.log('Results saved to:', resultsPath);
    } catch (error) {
      console.error('Error saving results file:', error);
      throw error;
    }

    return NextResponse.json({ 
      success: true, 
      testId,
      memoryScore,
      cognitiveAssessment: {
        averageScore: avgCognitiveScore,
        totalAnalyzed: cognitiveResults.length,
        results: cognitiveResults
      },
      savedLocation: userDir,
      filesCreated: {
        resultsFile: resultsFilename,
        audioFiles: audioFiles.length
      },
      message: `Kết quả đã được lưu thành công tại: ${userDir}`
    });

  } catch (error) {
    console.error('Error saving test results:', error);
    return NextResponse.json(
      { 
        error: 'Failed to save test results',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 });
    }

    // Đường dẫn tìm file
    const baseDir = join(process.cwd(), 'frontend', 'recordings');
    const userDir = baseDir;
    
    try {
      // Đọc danh sách file trong thư mục user
      const fs = require('fs').promises;
      const files = await fs.readdir(userDir);
      const jsonFiles = files.filter((file: string) => file.endsWith('.json'));
      
      const tests = [];
      for (const file of jsonFiles) {
        try {
          const filePath = join(userDir, file);
          const content = await fs.readFile(filePath, 'utf8');
          const testData = JSON.parse(content);
          tests.push({
            testId: testData.testId,
            completedAt: testData.completedAt,
            memoryScore: testData.memoryScore || calculateMemoryScore(testData.results),
            cognitiveScore: testData.cognitiveAssessment?.averageScore || 0,
            totalQuestions: testData.totalQuestions,
            answeredQuestions: testData.answeredQuestions,
            cognitiveAnalyzed: testData.cognitiveAssessment?.totalAnalyzed || 0
          });
        } catch (err) {
          console.error(`Error reading file ${file}:`, err);
        }
      }
      
      return NextResponse.json({ 
        tests: tests.sort((a, b) => new Date(b.completedAt).getTime() - new Date(a.completedAt).getTime()),
        location: userDir,
        message: 'Test history retrieved successfully'
      });
      
    } catch (error) {
      console.error('Error reading user directory:', error);
      return NextResponse.json({ 
        tests: [],
        error: 'Could not read test history',
        location: userDir
      });
    }

  } catch (error) {
    console.error('Error retrieving test results:', error);
    return NextResponse.json(
      { error: 'Failed to retrieve test results' },
      { status: 500 }
    );
  }
}

// Hàm phân tích cognitive assessment thông qua backend Python
async function analyzeCognitiveAssessment(
  audioFile: File,
  participantInfo: { age: number; gender: 'male' | 'female' | 'other' },
  transcribedText: string,
  userId?: string,
  question?: string,
  questionId?: number
): Promise<CognitiveAssessmentResult | null> {
  try {
    const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    
    const formData = new FormData();
    formData.append('audioFile', audioFile);
    formData.append('age', participantInfo.age.toString());
    formData.append('gender', participantInfo.gender);
    formData.append('transcribedText', transcribedText);
    if (userId) formData.append('userId', userId);
    if (question) formData.append('question', question);
    if (typeof questionId !== 'undefined') formData.append('questionId', String(questionId));

    const response = await fetch(`${PYTHON_BACKEND_URL}/assess-file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Cognitive assessment error:', errorText);
      return null;
    }

    const result = await response.json();
    return result;
    
  } catch (error) {
    console.error('Error calling Python backend for cognitive assessment:', error);
    return null;
  }
}

// Hàm tính điểm memory test
function calculateMemoryScore(results: TestResult[]): number {
  let totalScore = 0;
  const maxScore = 100;
  
  results.forEach(result => {
    if (result.transcription && result.transcription.trim().length > 0) {
      const wordCount = result.transcription.split(' ').length;
      const questionScore = Math.min(wordCount * 2, 10);
      totalScore += questionScore;
    }
  });

  const normalizedScore = Math.min((totalScore / results.length) * 10, maxScore);
  return Math.round(normalizedScore);
}