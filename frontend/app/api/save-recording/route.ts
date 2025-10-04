// app/api/save-recording/route.ts - Enhanced version with better error handling

import { NextRequest, NextResponse } from 'next/server';
import { mkdir, writeFile, access } from 'fs/promises';
import { join } from 'path';
import { constants } from 'fs';

// Ensure recordings directory exists
async function ensureRecordingsDirectory() {
  const recordingsDir = join(process.cwd(), 'recordings');
  try {
    await access(recordingsDir, constants.F_OK);
  } catch {
    // Directory doesn't exist, create it
    await mkdir(recordingsDir, { recursive: true });
    console.log('Created recordings directory:', recordingsDir);
  }
  return recordingsDir;
}

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

// Sanitize filename to prevent path traversal
function sanitizeFilename(filename: string): string {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_') // Replace invalid chars with underscore
    .replace(/_{2,}/g, '_') // Replace multiple underscores with single
    .substring(0, 200); // Limit length
}

export async function POST(request: NextRequest) {
  console.log('🎤 Save recording request received');
  
  try {
    const formData = await request.formData();
    const audioFile = formData.get('audio') as File;
    const sessionId = formData.get('sessionId') as string;
    const questionId = formData.get('questionId') as string;
    
    // Validate required fields
    if (!audioFile) {
      return NextResponse.json({ 
        error: 'No audio file provided',
        details: 'audio field is required' 
      }, { status: 400 });
    }
    
    if (!sessionId) {
      return NextResponse.json({ 
        error: 'Session ID is required',
        details: 'sessionId field is required' 
      }, { status: 400 });
    }
    
    if (!questionId) {
      return NextResponse.json({ 
        error: 'Question ID is required',
        details: 'questionId field is required' 
      }, { status: 400 });
    }

    // Validate file size (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (audioFile.size > maxSize) {
      return NextResponse.json({ 
        error: 'File too large',
        details: `Maximum file size is ${maxSize / (1024 * 1024)}MB, received ${Math.round(audioFile.size / (1024 * 1024))}MB`
      }, { status: 413 });
    }
    
    // Validate file is not empty
    if (audioFile.size === 0) {
      return NextResponse.json({ 
        error: 'Empty file',
        details: 'Audio file cannot be empty' 
      }, { status: 400 });
    }

    // Validate file type
    if (!validateAudioFile(audioFile.name, audioFile.type)) {
      return NextResponse.json({ 
        error: 'Invalid file type',
        details: 'Only audio files are allowed (webm, wav, mp3, mp4, ogg)',
        received: {
          filename: audioFile.name,
          contentType: audioFile.type
        }
      }, { status: 400 });
    }

    // Prepare file storage
    const recordingsDir = await ensureRecordingsDirectory();
    const sanitizedFilename = sanitizeFilename(audioFile.name);
    const filepath = join(recordingsDir, sanitizedFilename);
    
    // Kiểm tra quyền ghi file
    try {
      await writeFile(filepath + '.test', 'test');
      await access(filepath + '.test', constants.F_OK);
      await import('fs/promises').then(fs => fs.unlink(filepath + '.test'));
    } catch (e) {
      console.error('❌ Không có quyền ghi file vào recordings:', e);
      return NextResponse.json({
        success: false,
        error: 'Không có quyền ghi file vào recordings',
        details: e instanceof Error ? e.message : e
      }, { status: 500 });
    }

    console.log('📁 Saving to:', filepath);
    console.log('📊 File info:', {
      name: audioFile.name,
      size: audioFile.size,
      type: audioFile.type,
      sessionId,
      questionId
    });

    try {
      // Convert file to buffer
      const bytes = await audioFile.arrayBuffer();
      const buffer = Buffer.from(bytes);
      
      // Validate buffer is not empty
      if (buffer.length === 0) {
        throw new Error('File buffer is empty');
      }
      
      // Write file to disk
      await writeFile(filepath, buffer);
      
      // Verify file was written successfully
      try {
        await access(filepath, constants.F_OK);
      } catch {
        throw new Error('File was not saved successfully');
      }
      
      console.log('✅ Audio file saved successfully:', {
        filename: sanitizedFilename,
        size: buffer.length,
        path: filepath
      });
      
      // Return success response
      return NextResponse.json({
        success: true,
        message: 'Audio file saved successfully',
        filename: sanitizedFilename,
        size: buffer.length,
        sessionId,
        questionId,
        timestamp: new Date().toISOString()
      });
      
    } catch (fileError) {
      console.error('❌ File operations failed:', fileError);
      throw new Error(`Failed to save file: ${fileError instanceof Error ? fileError.message : 'Unknown error'}`);
    }
    
  } catch (error) {
    console.error('❌ Save recording API error:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    const errorDetails = {
      timestamp: new Date().toISOString(),
      error: errorMessage,
    };
    
    // Determine appropriate status code
    let statusCode = 500;
    if (errorMessage.includes('File too large') || errorMessage.includes('Maximum file size')) {
      statusCode = 413;
    } else if (errorMessage.includes('required') || errorMessage.includes('Invalid')) {
      statusCode = 400;
    }
    
    return NextResponse.json(
      { 
        success: false,
        details: errorMessage,
        ...errorDetails,
        troubleshoot: {
          step1: 'Check if the recordings directory exists and is writable',
          step2: 'Verify the audio file is valid and not corrupted',
          step3: 'Ensure sufficient disk space is available',
          step4: 'Check file permissions for the application directory'
        }
      },
      { status: statusCode }
    );
  }
}

// GET endpoint to check service status
export async function GET() {
  try {
    const recordingsDir = await ensureRecordingsDirectory();
    
    // Check if directory is writable
    const testFile = join(recordingsDir, 'test.tmp');
    try {
      await writeFile(testFile, 'test');
      await access(testFile, constants.F_OK);
      // Clean up test file
      await import('fs/promises').then(fs => fs.unlink(testFile));
    } catch {
      throw new Error('Directory is not writable');
    }
    
    return NextResponse.json({
      status: 'ready',
      message: 'Save recording service is operational',
      recordingsDirectory: recordingsDir,
      maxFileSize: '50MB',
      supportedFormats: ['webm', 'wav', 'mp3', 'mp4', 'ogg'],
      endpoints: {
        'POST /api/save-recording': 'Save audio recording',
        'GET /api/save-recording': 'Check service status'
      }
    });
  } catch (error) {
    return NextResponse.json({
      status: 'error',
      message: 'Save recording service is not operational',
      error: error instanceof Error ? error.message : 'Unknown error',
      troubleshoot: {
        step1: 'Check if recordings directory exists',
        step2: 'Verify write permissions for the application',
        step3: 'Ensure sufficient disk space'
      }
    }, { status: 500 });
  }
}

// OPTIONS handler for CORS
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}