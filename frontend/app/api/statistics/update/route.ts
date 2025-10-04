import { NextRequest, NextResponse } from 'next/server';

const inMemoryStats: Array<{
  sessionId: string;
  score: number;
  completionTime?: number;
  audioQuality?: number;
  domain: string;
  createdAt: string;
}> = [];

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const record = {
      sessionId: String(body.sessionId || ''),
      score: Number(body.score || 0),
      completionTime: typeof body.completionTime === 'number' ? body.completionTime : undefined,
      audioQuality: typeof body.audioQuality === 'number' ? body.audioQuality : undefined,
      domain: String(body.domain || 'MMSE'),
      createdAt: new Date().toISOString(),
    };
    inMemoryStats.push(record);
    return NextResponse.json({ success: true });
  } catch (e) {
    return NextResponse.json({ success: false, error: 'Invalid payload' }, { status: 400 });
  }
}

export async function GET() {
  return NextResponse.json({ success: true, stats: inMemoryStats });
}


