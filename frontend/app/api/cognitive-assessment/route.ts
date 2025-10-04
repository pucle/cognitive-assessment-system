import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    const audioFile = formData.get('audio') as File;
    const age = formData.get('age') as string;
    const gender = formData.get('gender') as string;
    const transcript = formData.get('transcript') as string;
    const question = formData.get('question') as string;
    const user_id = formData.get('user_id') as string;

    if (!transcript) {
      return NextResponse.json({ 
        error: 'Missing required field: transcript' 
      }, { status: 400 });
    }

    // Forward request to Python backend
    const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    
    const backendFormData = new FormData();
    if (audioFile) {
      backendFormData.append('audio', audioFile);
    }
    backendFormData.append('transcript', transcript);
    backendFormData.append('question', question || '');
    backendFormData.append('user_id', user_id || 'anonymous');
    backendFormData.append('age', age || '65');
    backendFormData.append('gender', gender || 'unknown');

    console.log(`Forwarding request to Python backend: ${PYTHON_BACKEND_URL}/assess-cognitive`);
    
    const backendResponse = await fetch(`${PYTHON_BACKEND_URL}/assess-cognitive`, {
      method: 'POST',
      body: backendFormData,
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error('Python backend error:', errorText);
      
      let errorMessage = `Backend error: ${backendResponse.status}`;
      try {
        const errorData = JSON.parse(errorText);
        errorMessage = errorData.error || errorData.message || errorMessage;
      } catch {
        errorMessage = errorText || errorMessage;
      }
      
      return NextResponse.json({ 
        error: errorMessage,
        details: 'Failed to process cognitive assessment'
      }, { status: 500 });
    }

    const result = await backendResponse.json();
    return NextResponse.json(result);

  } catch (error) {
    console.error('Error in cognitive assessment API:', error);
    return NextResponse.json({ 
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Cognitive Assessment API',
    version: '1.0.0',
    endpoints: {
      POST: '/api/cognitive-assessment'
    }
  });
}
