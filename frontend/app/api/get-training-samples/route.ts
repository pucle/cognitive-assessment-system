import { NextRequest, NextResponse } from 'next/server';
import db from '@/db/drizzle';
import { sql } from 'drizzle-orm';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');
    const usageMode = searchParams.get('usageMode') || 'personal';
    const userEmail = searchParams.get('userEmail');
    const limit = parseInt(searchParams.get('limit') || '50');

    console.log('🔍 Fetching training samples with db.execute...');
    console.log(`📋 Params: userId=${userId}, usageMode=${usageMode}, userEmail=${userEmail}, limit=${limit}`);

    let results;

    try {
      if (usageMode === 'personal' && userId) {
        console.log(`👤 Fetching personal training samples for userId: ${userId}`);
        results = await db.execute(sql`
          SELECT * FROM training_samples
          WHERE user_id = ${userId}
          ORDER BY created_at DESC
          LIMIT ${limit}
        `);
      } else if (usageMode === 'community' && userEmail) {
        console.log(`🌐 Fetching community training samples for email: ${userEmail}`);
        results = await db.execute(sql`
          SELECT * FROM training_samples
          WHERE user_email = ${userEmail}
          ORDER BY created_at DESC
          LIMIT ${limit}
        `);
      } else {
        console.log(`📊 Fetching recent training samples (no specific filter)`);
        results = await db.execute(sql`
          SELECT * FROM training_samples
          ORDER BY created_at DESC
          LIMIT ${limit}
        `);
      }

      console.log(`✅ Database query successful, found ${results.rows?.length || 0} results`);

      // Transform data to match expected format
      const transformedResults = (results.rows || []).map(item => ({
        id: item.id,
        sessionId: item.session_id,
        userId: item.user_id,
        userEmail: item.user_email,
        userName: item.user_name,
        questionId: item.question_id,
        questionText: item.question_text,
        audioFilename: item.audio_filename,
        audioUrl: item.audio_url,
        autoTranscript: item.auto_transcript,
        manualTranscript: item.manual_transcript,
        createdAt: item.created_at,
        updatedAt: item.updated_at,
      }));

      return NextResponse.json({
        success: true,
        count: transformedResults.length,
        data: transformedResults,
        mode: usageMode,
        userId: userId,
        userEmail: userEmail,
        debug: {
          searchParams: Object.fromEntries(searchParams.entries()),
          query: 'SELECT from training_samples using db.execute'
        }
      });

    } catch (dbError) {
      console.error('❌ Database query error:', dbError);

      // Return helpful error info
      const error = dbError instanceof Error ? dbError : new Error('Unknown database error');
      return NextResponse.json({
        success: false,
        error: 'Database query failed',
        details: {
          message: error.message,
          code: (error as any).code,
          hint: 'Table training_samples may not exist or columns may not match expected schema'
        },
        expectedColumns: [
          'id', 'session_id', 'user_id', 'user_email', 'user_name',
          'question_id', 'question_text', 'audio_filename', 'audio_url',
          'auto_transcript', 'manual_transcript', 'created_at', 'updated_at'
        ]
      }, { status: 500 });
    }

  } catch (error) {
    console.error('❌ Error retrieving training samples:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

