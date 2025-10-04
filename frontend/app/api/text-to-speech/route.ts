// app/api/text-to-speech/route.ts - Google TTS (gTTS)
import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { promisify } from 'util';
import { exec } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';

const execAsync = promisify(exec);

// Ensure Node.js runtime for child_process on Next.js
export const runtime = 'nodejs';

const GTTS_LANGUAGES = {
  'vi': 'vi',     
  'en': 'en',     
  'ja': 'ja',     
  'ko': 'ko',     
  'zh': 'zh',  
  'fr': 'fr',    
  'de': 'de',    
  'es': 'es',    
} as const;

// Speed settings
const SPEED_PRESETS = {
  'slow': false,
  'normal': true,
  'fast': false, 
} as const;

//definitions
interface GTTSCheckSuccess {
  available: true;
  message: string;
  type: 'direct' | 'python';
  cmd: string;
  python?: string;
}

interface GTTSCheckFailure {
  available: false;
  message: string;
  suggestion: string;
}

type GTTSCheckResult = GTTSCheckSuccess | GTTSCheckFailure;


function getPythonCommand(gttsInfo: GTTSCheckSuccess): string {
  if (gttsInfo.type === 'python') {
    return gttsInfo.python || 'python';
  }
  return 'gtts-cli';
}


async function checkGTTS(): Promise<GTTSCheckResult> {
  const commands = [
    { cmd: 'gtts-cli --help', type: 'direct' as const },
    { cmd: 'python -m gtts.cli --help', type: 'python' as const, python: 'python' },
    { cmd: 'python3 -m gtts.cli --help', type: 'python' as const, python: 'python3' },
    { cmd: 'py -m gtts.cli --help', type: 'python' as const, python: 'py' } 
  ];

  for (const command of commands) {
    try {
      await execAsync(command.cmd);
      console.log(`gTTS available via: ${command.cmd}`);
      return { 
        available: true, 
        message: `gTTS available via ${command.type}`,
        type: command.type,
        cmd: command.cmd,
        python: command.python
      };
    } catch {
      console.log(`Failed: ${command.cmd}`);
      continue;
    }
  }

  return { 
    available: false, 
    message: 'gTTS not found in any format',
    suggestion: 'Install with: pip install gTTS or pip3 install gTTS'
  };
}

async function generateSpeechFile(text: string, language: string, slow: boolean, gttsInfo: GTTSCheckSuccess) {
  const tempDir = os.tmpdir();
  const timestamp = typeof window !== 'undefined' ? Date.now() : Math.floor(Math.random() * 1000000);
  const randomId = typeof window !== 'undefined' ? Math.random().toString(36).substr(2, 9) : 'server';
  const filename = `gtts_${timestamp}_${randomId}.mp3`;
  const outputPath = path.join(tempDir, filename);

  const command = getPythonCommand(gttsInfo);
  let args: string[];

  // Use positional args and short flags to avoid Windows quoting issues
  if (gttsInfo.type === 'python') {
    // Set cwd to temp and use filename to avoid absolute path quirks on Windows
    args = ['-m', 'gtts.cli', text, '-l', language, '-o', filename];
    if (slow) args.push('-s');
  } else {
    args = [text, '-l', language, '-o', filename];
    if (slow) args.push('-s');
  }

  console.log(`🎵 Executing (file): ${command} [text] -l ${language} -o ${filename}${slow ? ' -s' : ''}`);

  return new Promise<Buffer>((resolve, reject) => {
    const errors: string[] = [];
    const gttsProcess = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'], shell: false, cwd: tempDir });

    gttsProcess.stderr.on('data', (data: Buffer) => {
      errors.push(data.toString());
    });

    const finalize = async (ok: boolean) => {
      try {
        if (!ok) {
          return reject(new Error(`gTTS failed: ${errors.join('\n') || 'Unknown error'}`));
        }
        const stats = await fs.stat(outputPath);
        if (stats.size < 500) {
          await fs.unlink(outputPath).catch(() => {});
          return reject(new Error('Generated audio file is too small'));
        }
        const audioBuffer = await fs.readFile(outputPath);
        await fs.unlink(outputPath).catch(() => {});
        console.log(`✅ Generated audio file: ${stats.size} bytes`);
        resolve(audioBuffer);
      } catch (e) {
        reject(new Error(`Audio file not generated: ${e instanceof Error ? e.message : 'Unknown error'}`));
      }
    };

    const timeout = setTimeout(() => {
      gttsProcess.kill('SIGKILL');
      finalize(false);
    }, 30000);

    gttsProcess.on('close', (code) => {
      clearTimeout(timeout);
      finalize(code === 0);
    });

    gttsProcess.on('error', () => {
      clearTimeout(timeout);
      finalize(false);
    });
  });
}

// Generate speech using stdout approach (fallback for gTTS)
async function generateSpeechStream(text: string, language: string, slow: boolean, gttsInfo: GTTSCheckSuccess): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const errors: string[] = [];
    
    const command = getPythonCommand(gttsInfo);
    let args: string[];
    
    if (gttsInfo.type === 'python') {
      args = ['-m', 'gtts.cli', text, '-l', language, '-o', '-'];
      if (slow) args.push('-s');
    } else {
      args = [text, '-l', language, '-o', '-'];
      if (slow) args.push('-s');
    }

    console.log(`🎵 Streaming: ${command} [text] -l ${language} -o -${slow ? ' -s' : ''}`);
    
    // Spawn process
    const gttsProcess = spawn(command, args, {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Collect audio data
    gttsProcess.stdout.on('data', (chunk: Buffer) => {
      chunks.push(chunk);
    });
    
    // Collect errors
    gttsProcess.stderr.on('data', (data: Buffer) => {
      const errorMsg = data.toString();
      errors.push(errorMsg);
    });
    
    // Handle process completion
    gttsProcess.on('close', (code: number) => {
      if (code === 0 && chunks.length > 0) {
        const audioBuffer = Buffer.concat(chunks);
        
        if (audioBuffer.length < 500) {
          reject(new Error('Generated audio is too small or invalid'));
          return;
        }
        
        console.log(`✅ Stream success: ${audioBuffer.length} bytes`);
        resolve(audioBuffer);
      } else {
        const errorMessage = errors.length > 0 ? errors.join(', ') : `Process exited with code ${code}`;
        reject(new Error(`gTTS failed: ${errorMessage}`));
      }
    });
    
    // Handle spawn errors
    gttsProcess.on('error', (error: Error) => {
      reject(new Error(`Failed to spawn gTTS: ${error.message}`));
    });
    
    // Set timeout
    const timeout = setTimeout(() => {
      gttsProcess.kill('SIGKILL');
      reject(new Error('TTS request timeout (30 seconds)'));
    }, 30000);
    
    gttsProcess.on('close', () => {
      clearTimeout(timeout);
    });
  });
}

export async function POST(request: NextRequest) {
  console.log('🎤 gTTS Request received');
  
  try {
    const { 
      text, 
      language = 'vi',
      speed = 'normal'
    } = await request.json();

    console.log('Request params:', { 
      textLength: text?.length, 
      language, 
      speed,
      textPreview: text?.substring(0, 100) 
    });

    // Validate input
    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Text is required' }, { status: 400 });
    }

    if (text.trim().length === 0) {
      return NextResponse.json({ error: 'Text cannot be empty' }, { status: 400 });
    }

    if (text.length > 5000) {
      return NextResponse.json({ 
        error: 'Text too long. Maximum 5000 characters allowed.' 
      }, { status: 400 });
    }

    // Clean text for better pronunciation
    const cleanText = text
      .replace(/[{}]/g, '') // Remove curly braces
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();

    // Check if gTTS is available
    console.log('🔍 Checking gTTS availability...');
    const gttsCheck = await checkGTTS();
    
    if (!gttsCheck.available) {
      console.error('❌ gTTS not available:', gttsCheck.message);
      return NextResponse.json({ 
        error: 'gTTS not available',
        details: gttsCheck.message,
        suggestion: gttsCheck.suggestion
      }, { status: 500 });
    }

    console.log('✅ gTTS available:', gttsCheck.message);

    // Type guard to ensure we have a successful check
    if (!gttsCheck.available) {
      throw new Error('gTTS check should have failed earlier');
    }

    // Select language and speed settings
    const selectedLanguage = GTTS_LANGUAGES[language as keyof typeof GTTS_LANGUAGES] || GTTS_LANGUAGES.vi;
    const isSlowSpeed = speed === 'slow';

    console.log(`🎵 Using language: ${selectedLanguage}, slow: ${isSlowSpeed}`);

    let audioBuffer: Buffer;

    try {
      // Try file-based approach first (more reliable)
      console.log('🎵 Attempting file-based generation...');
      audioBuffer = await generateSpeechFile(cleanText, selectedLanguage, isSlowSpeed, gttsCheck);
    } catch (fileError) {
      console.log('⚠️ File-based generation failed, trying stream approach:', fileError);
      
      try {
        // Fallback to stream approach
        audioBuffer = await generateSpeechStream(cleanText, selectedLanguage, isSlowSpeed, gttsCheck);
      } catch (streamError) {
        console.error('❌ Both approaches failed');
        throw new Error(`gTTS generation failed: ${streamError instanceof Error ? streamError.message : 'Unknown error'}`);
      }
    }

    console.log(`🎉 gTTS Success: Generated ${audioBuffer.length} bytes`);
    
    // Convert Buffer to Uint8Array for NextResponse
    const audioData = new Uint8Array(audioBuffer);
    
    return new NextResponse(audioData, {
      status: 200,
      headers: {
        'Content-Type': 'audio/mpeg',
        'Content-Length': audioBuffer.length.toString(),
        'Cache-Control': 'public, max-age=3600',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
    
  } catch (error) {
    console.error('❌ gTTS API Error:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { 
        error: 'Failed to generate speech',
        details: errorMessage,
        timestamp: new Date().toISOString(),
        troubleshoot: {
          step1: 'Check if gTTS is installed: pip install gTTS',
          step2: 'Test manually: gtts-cli --text "xin chào" --lang vi --output test.mp3',
          step3: 'Check if Python is in PATH: python --version',
          step4: 'Try different installation: pip3 install gTTS',
          step5: 'For Windows: py -m pip install gTTS'
        }
      },
      { status: 500 }
    );
  }
}

// GET endpoint to list available languages, test gTTS, and provide diagnostics
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const test = searchParams.get('test');
  
  if (test === 'true') {
    // Test gTTS functionality
    try {
      const gttsCheck = await checkGTTS();
      const testText = "Xin chào, đây là bài kiểm tra.";
      
      if (gttsCheck.available) {
        // Try to generate a small test audio
        try {
          const audioBuffer = await generateSpeechFile(testText, 'vi', false, gttsCheck);
          
          return NextResponse.json({
            status: 'success',
            gttsAvailable: true,
            testAudioSize: audioBuffer.length,
            checkResult: gttsCheck.message,
            commandType: gttsCheck.type
          });
        } catch (testError) {
          // Fallback to streaming test
          try {
            const audioBuffer = await generateSpeechStream(testText, 'vi', false, gttsCheck);
            return NextResponse.json({
              status: 'success',
              gttsAvailable: true,
              testAudioSize: audioBuffer.length,
              checkResult: gttsCheck.message,
              commandType: gttsCheck.type,
              fallback: 'stream'
            });
          } catch (streamErr) {
            return NextResponse.json({
              status: 'error',
              gttsAvailable: true,
              error: testError instanceof Error ? testError.message : 'Unknown error',
              streamError: streamErr instanceof Error ? streamErr.message : 'Unknown stream error',
              checkResult: gttsCheck.message,
              commandType: gttsCheck.type,
              issue: 'gTTS available but both file and stream generation failed'
            });
          }
        }
      } else {
        return NextResponse.json({
          status: 'error',
          gttsAvailable: false,
          issue: gttsCheck.message,
          suggestion: gttsCheck.suggestion
        });
      }
    } catch (error) {
      return NextResponse.json({
        status: 'error',
        message: 'Failed to test gTTS',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Regular GET response
  return NextResponse.json({
    service: 'Google Text-to-Speech (gTTS) API',
    languages: Object.keys(GTTS_LANGUAGES),
    languageDetails: GTTS_LANGUAGES,
    speedOptions: Object.keys(SPEED_PRESETS),
    usage: {
      text: 'Text to convert to speech (max 5000 characters)',
      language: 'Language code: vi (Vietnamese), en (English), ja (Japanese), etc.',
      speed: 'Speech speed: normal, slow'
    },
    endpoints: {
      'POST /api/text-to-speech': 'Generate speech from text',
      'GET /api/text-to-speech': 'Get API info and available options',
      'GET /api/text-to-speech?test=true': 'Test gTTS functionality'
    },
    installation: {
      command: 'pip install gTTS',
      alternatives: ['pip3 install gTTS', 'py -m pip install gTTS'],
      documentation: 'https://gtts.readthedocs.io/'
    }
  });
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