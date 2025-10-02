import { put } from '@vercel/blob';
import { NextRequest, NextResponse } from 'next/server';
import { auth, currentUser } from "@clerk/nextjs/server";

export const runtime = 'nodejs';

// Validate file extension and MIME type
function validateAudioFile(filename: string, contentType: string | null): boolean {
  const allowedExtensions = ['.webm', '.wav', '.mp3', '.mp4', '.ogg'];
  const allowedMimeTypes = [
    'audio/webm',
    'audio/wav', 
    'audio/mpeg',
    'audio/mp4',
    'audio/ogg'
  ];
  
  const hasValidExtension = allowedExtensions.some(ext => 
    filename.toLowerCase().endsWith(ext)
  );
  
  const hasValidMimeType = !contentType || allowedMimeTypes.some(type =>
    contentType.includes(type.split('/')[1])
  );
  
  return hasValidExtension && hasValidMimeType;
}

// Sanitize filename to prevent issues
function sanitizeFilename(filename: string): string {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .replace(/_{2,}/g, '_')
    .substring(0, 200);
}

export async function POST(request: NextRequest) {
  try {
    console.log('🎵 Save recording blob API called');

    // Check authentication
    let userId: string | null = null;
    let user: any = null;
    let userEmail: string = 'demo@example.com';

    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

    if (isClerkAvailable) {
      try {
        const authResult = await auth();
        userId = authResult.userId;
        user = await currentUser();
        userEmail = user?.primaryEmailAddress?.emailAddress || userEmail;
      } catch (authError) {
        console.error("Auth error:", authError);
        return NextResponse.json({ error: "Authentication failed" }, { status: 401 });
      }

      if (!userId) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
    } else {
      console.warn('Save-recording-blob API: Clerk not available, using demo mode');
      userId = 'demo-user';
    }

    // Parse multipart form data
    const formData = await request.formData();
    const audioFile = formData.get('recording') as File;
    const sessionId = formData.get('sessionId') as string;
    const questionId = formData.get('questionId') as string;

    // Validation
    if (!audioFile) {
      return NextResponse.json({
        success: false,
        error: 'No audio file provided'
      }, { status: 400 });
    }

    if (!sessionId) {
      return NextResponse.json({
        success: false,
        error: 'Session ID is required'
      }, { status: 400 });
    }

    // Validate file type
    if (!validateAudioFile(audioFile.name, audioFile.type)) {
      return NextResponse.json({
        success: false,
        error: `Invalid file type. Allowed: .webm, .wav, .mp3, .mp4, .ogg`
      }, { status: 400 });
    }

    // Create unique filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const sanitizedFileName = sanitizeFilename(audioFile.name) || 'recording.webm';
    const fileName = `recordings/${userEmail}/${sessionId}/${questionId || 'general'}_${timestamp}_${sanitizedFileName}`;

    console.log('📁 Uploading to Vercel Blob:', {
      fileName,
      size: audioFile.size,
      type: audioFile.type,
      userEmail,
      sessionId
    });

    // Upload to Vercel Blob
    const blob = await put(fileName, audioFile, {
      access: 'public',
      addRandomSuffix: false,
    });

    const responseData = {
      success: true,
      message: 'Recording saved successfully to Vercel Blob',
      data: {
        blobUrl: blob.url,
        fileName: fileName,
        size: audioFile.size,
        type: audioFile.type,
        sessionId,
        questionId,
        timestamp: new Date().toISOString(),
        userEmail
      }
    };

    console.log('✅ Recording saved successfully:', responseData.data);

    return NextResponse.json(responseData);

  } catch (error) {
    console.error('❌ Error saving recording to Vercel Blob:', error);

    // Determine if it's a client error or server error
    const isClientError = error instanceof Error && 
      (error.message.includes('Invalid file') || 
       error.message.includes('required') ||
       error.message.includes('Session ID'));

    return NextResponse.json({
      success: false,
      error: 'Failed to save recording',
      details: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { 
      status: isClientError ? 400 : 500 
    });
  }
}
