import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Accept both 'audioFile' and 'audio' to be compatible with different callers
    const audioFile = (formData.get('audioFile') as File) || (formData.get('audio') as File);
    const question = formData.get('question') as string;
    const age = formData.get('age') as string;
    const gender = formData.get('gender') as string;
    const user = formData.get('user') as string;
    const questionId = formData.get('questionId') as string;
    const language = formData.get('language') as string || 'vi';
    const transcript = (formData.get('transcript') as string) || '';

    if (!audioFile) {
      return NextResponse.json({ 
        error: 'No audio file provided' 
      }, { status: 400 });
    }

    // Forward request to Python backend (prefer assess if transcript provided)
    const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

    const backendFormData = new FormData();
    backendFormData.append('audio', audioFile);
    backendFormData.append('question', question || '');
    backendFormData.append('age', age || '65');
    backendFormData.append('gender', gender || 'unknown');
    backendFormData.append('user', user || 'anonymous');
    backendFormData.append('questionId', questionId || '');
    backendFormData.append('language', language);
    if (transcript && transcript.trim().length > 0) {
      backendFormData.append('transcript', transcript);
    }

    const endpoint = transcript && transcript.trim().length > 0 ? '/api/assess' : '/auto-transcribe';
    console.log(`Forwarding audio analysis to Python backend: ${PYTHON_BACKEND_URL}${endpoint}`);

    const backendResponse = await fetch(`${PYTHON_BACKEND_URL}${endpoint}`, {
      method: 'POST',
      body: backendFormData,
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error('Python backend transcription error:', errorText);
      
      return NextResponse.json({ 
        success: false,
        error: `Transcription failed: ${backendResponse.status}`,
        data: {
          transcript: '',
          method: 'failed',
          cost: 0
        }
      }, { status: 500 });
    }

    const assessmentResult = await backendResponse.json();
    
    if (!assessmentResult.success) {
      return NextResponse.json({ 
        success: false,
        error: assessmentResult.error || 'Assessment failed',
        data: {
          transcript: '',
          method: 'failed',
          cost: 0
        }
      }, { status: 500 });
    }

    // Extract data from the comprehensive assessment result
    const transcriptionData = assessmentResult.transcription || {};
    const audioFeatures = assessmentResult.audio_features || {};
    const mlPrediction = assessmentResult.ml_prediction || {};
    const gptEvaluation = assessmentResult.gpt_evaluation || {};

    // Debug logging
    console.log('Backend response:', assessmentResult);
    console.log('Transcription data:', transcriptionData);
    console.log('Audio features:', audioFeatures);
    console.log('ML prediction:', mlPrediction);
    console.log('GPT evaluation:', gptEvaluation);

    return NextResponse.json({
      success: true,
      data: {
        transcript: (transcriptionData.transcript && transcriptionData.transcript.trim().length > 0) ? transcriptionData.transcript : 'Không có lời thoại',
        confidence: typeof transcriptionData.confidence === 'number' && isFinite(transcriptionData.confidence) ? transcriptionData.confidence : 0,
        model: transcriptionData.model || 'openai-whisper-1',
        method: transcriptionData.model || 'openai-whisper-1',
        cost: 0,
        audio_features: audioFeatures,
        ml_prediction: mlPrediction,
        gpt_evaluation: gptEvaluation,
        final_score: typeof assessmentResult.final_score === 'number' && isFinite(assessmentResult.final_score) ? assessmentResult.final_score : 0,
        language: assessmentResult.language || 'vi',
        textRecord: {
          filename: audioFile.name,
          size: audioFile.size,
          type: audioFile.type
        }
      }
    });

  } catch (error) {
    console.error('Error in analyze-audio API:', error);
    return NextResponse.json({ 
      success: false,
      error: 'Internal server error',
      data: {
        transcript: '',
        method: 'error',
        cost: 0
      }
    }, { status: 500 });
  }
}
