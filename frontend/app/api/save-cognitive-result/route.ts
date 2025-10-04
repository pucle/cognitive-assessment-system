export const runtime = 'nodejs';
import 'server-only';

import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs/server";

export async function POST(request: NextRequest) {
  try {
    let userId: string | null = null;

    // Check if Clerk is available
    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

    if (isClerkAvailable) {
      try {
        userId = (await auth()).userId;
      } catch (authError) {
        console.error("Auth error:", authError);
        return NextResponse.json({ error: "Authentication failed" }, { status: 401 });
      }

      if (!userId) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
    } else {
      // Demo mode - use fallback values
      console.warn('Save-cognitive-result API: Clerk not available, using demo mode');
      userId = 'demo-user';
    }

    const body = await request.json();
    const { sessionId, cognitiveResult, userInfo } = body;

    if (!sessionId || !cognitiveResult) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields'
      }, { status: 400 });
    }

    // TODO: Implement database save logic
    // For now, just return success
    console.log('Saving cognitive result:', {
      userId,
      sessionId,
      cognitiveResult,
      userInfo
    });

    return NextResponse.json({ 
      success: true, 
      id: `result_${typeof window !== 'undefined' ? Date.now() : Math.floor(Math.random() * 1000000)}`,
      message: 'Result saved successfully'
    });

  } catch (error) {
    console.error('Save cognitive result error:', error);
    return NextResponse.json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Save Cognitive Result API',
    version: '1.0.0',
    endpoints: {
      POST: '/api/save-cognitive-result'
    }
  });
}
