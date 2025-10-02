import { put } from '@vercel/blob';
import { NextRequest, NextResponse } from 'next/server';
import { auth, currentUser } from "@clerk/nextjs/server";

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    let userId: string | null = null;
    let user: any = null;

    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

    if (isClerkAvailable) {
      try {
        const authResult = await auth();
        userId = authResult.userId;
        user = await currentUser();
      } catch (authError) {
        console.error("Auth error:", authError);
        return NextResponse.json({ error: "Authentication failed" }, { status: 401 });
      }

      if (!userId) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
    } else {
      // Demo mode
      userId = 'demo-user';
    }

    const formData = await request.formData();
    const file = formData.get('audio') as File;
    const sessionId = formData.get('sessionId') as string;
    const questionId = formData.get('questionId') as string;

    if (!file) {
      return NextResponse.json({ error: 'No audio file provided' }, { status: 400 });
    }

    if (!sessionId || !questionId) {
      return NextResponse.json({ error: 'Missing sessionId or questionId' }, { status: 400 });
    }

    // Create unique filename
    const timestamp = Date.now();
    const userEmail = user?.primaryEmailAddress?.emailAddress || 'demo@example.com';
    const fileName = `mmse/${userEmail}/${sessionId}/${questionId}_${timestamp}.webm`;

    // Upload to Vercel Blob
    const blob = await put(fileName, file, {
      access: 'public',
      addRandomSuffix: false,
    });

    console.log('Audio uploaded successfully:', {
      fileName,
      blobUrl: blob.url,
      size: file.size,
    });

    return NextResponse.json({ 
      success: true,
      blobUrl: blob.url,
      fileName: fileName,
      size: file.size,
      message: 'Audio uploaded successfully'
    });

  } catch (error) {
    console.error('Error uploading audio to Vercel Blob:', error);

    return NextResponse.json(
      { 
        success: false,
        error: 'Failed to upload audio',
        details: error instanceof Error ? error.message : 'Unknown error'
      }, 
      { status: 500 }
    );
  }
}
