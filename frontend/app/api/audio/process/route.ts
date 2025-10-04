import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const audio = formData.get('audio') as File | null;
    const questionId = (formData.get('questionId') as string) || '';
    const sessionId = (formData.get('sessionId') as string) || '';

    if (!audio) {
      return NextResponse.json({ success: false, error: 'No audio provided' }, { status: 400 });
    }

    // 1) Forward to internal analyze-audio (which proxies to Python backend)
    const forward = new FormData();
    forward.append('audio', audio, (audio as any)?.name || `recording_${questionId || 'NA'}.webm`);
    forward.append('questionId', questionId);
    forward.append('language', 'vi');

    const res = await fetch(`${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/analyze-audio`, {
      method: 'POST',
      body: forward,
    });

    if (!res.ok) {
      const txt = await res.text();
      return NextResponse.json({ success: false, error: `Analyze failed: ${txt}` }, { status: 502 });
    }

    const data = await res.json();
    if (!data?.success) {
      return NextResponse.json({ success: false, error: data?.error || 'Analyze failed' }, { status: 500 });
    }

    // 2) Optionally evaluate content (non-blocking fallback)
    const confidence = Number(data?.data?.confidence || 0) || 0;
    const transcript = String(data?.data?.transcript || '') || '';

    // Best-effort evaluation call; ignore failure
    try {
      await fetch(`${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/gpt/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ responseText: transcript, language: 'vi' }),
      });
    } catch {}

    // Return compact payload for the recorder component
    return NextResponse.json({
      success: true,
      tempId: `${Date.now()}_${Math.random().toString(36).slice(2)}`,
      transcript,
      audioPath: '',
      confidence,
    });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error?.message || 'Internal error' }, { status: 500 });
  }
}


