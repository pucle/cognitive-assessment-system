import { NextRequest, NextResponse } from 'next/server';
import { neon } from '@neondatabase/serverless';
import { desc, eq, sql, and } from 'drizzle-orm';
import { cognitiveAssessmentResults } from '@/db/schema';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '50');
    const sessionId = searchParams.get('sessionId');
    const userId = searchParams.get('userId');
    const usageMode = searchParams.get('usageMode');

    const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || 'file:./cognitive_assessment.db';

    console.log('🔍 Fetching cognitive assessment results from database...');

    let results;
    if (databaseUrl.startsWith('file:')) {
      // SQLite database
      console.log('📱 Using SQLite database');
      const { default: Database } = await import('better-sqlite3');
      const { drizzle: drizzleSQLite } = await import('drizzle-orm/better-sqlite3');

      const sqlite = new Database(databaseUrl.replace('file:', ''));
      const db = drizzleSQLite(sqlite);

      try {
        let query = db.select()
          .from(cognitiveAssessmentResults)
          .orderBy(desc(cognitiveAssessmentResults.createdAt));

        // Apply filters if provided - avoid circular references by building conditions properly
        if (sessionId) {
          query = query.where(eq(cognitiveAssessmentResults.sessionId, sessionId));
        }
        if (userId) {
          query = query.where(eq(cognitiveAssessmentResults.userId, userId));
        }
        if (usageMode) {
          query = query.where(eq(cognitiveAssessmentResults.usageMode, usageMode));
        }

        results = await query.limit(limit);

        // Process results to parse userInfo JSON
        results = results.map(result => {
          let userInfo = { name: 'N/A', email: 'N/A', age: 'N/A', gender: 'N/A' };

          if (result.userInfo) {
            try {
              // Parse JSON string from database
              if (typeof result.userInfo === 'string') {
                userInfo = { ...userInfo, ...JSON.parse(result.userInfo) };
              } else if (typeof result.userInfo === 'object') {
                userInfo = { ...userInfo, ...result.userInfo };
              }
            } catch (e) {
              console.warn('Failed to parse userInfo JSON:', e);
            }
          }

          return {
            ...result,
            userInfo
          };
        });

        console.log(`✅ SQLite query successful, found ${results.length} results`);
      } catch (sqliteError) {
        console.error('❌ SQLite query error:', sqliteError);
        throw sqliteError;
      } finally {
        sqlite.close();
      }
    } else {
      // PostgreSQL database
      console.log('🐘 Using PostgreSQL database');
      const sql = neon(databaseUrl);
      const { drizzle } = await import('drizzle-orm/neon-http');
      const db = drizzle(sql);

      let query = db.select()
        .from(cognitiveAssessmentResults)
        .orderBy(desc(cognitiveAssessmentResults.createdAt));

      // Apply filters if provided - avoid circular references by building conditions properly
      if (sessionId) {
        query = query.where(eq(cognitiveAssessmentResults.sessionId, sessionId));
      }
      if (userId) {
        query = query.where(eq(cognitiveAssessmentResults.userId, userId));
      }
      if (usageMode) {
        query = query.where(eq(cognitiveAssessmentResults.usageMode, usageMode));
      }

      results = await query.limit(limit);

      // Process results to parse userInfo JSON
      results = results.map(result => {
        let userInfo = { name: 'N/A', email: 'N/A', age: 'N/A', gender: 'N/A' };

        if (result.userInfo) {
          try {
            // Parse JSON from database
            if (typeof result.userInfo === 'string') {
              userInfo = { ...userInfo, ...JSON.parse(result.userInfo) };
            } else if (typeof result.userInfo === 'object') {
              userInfo = { ...userInfo, ...result.userInfo };
            }
          } catch (e) {
            console.warn('Failed to parse userInfo JSON:', e);
          }
        }

        return {
          ...result,
          userInfo
        };
      });

      console.log(`✅ PostgreSQL query successful, found ${results.length} results`);
    }

    // Log first result for debugging
    if (results && results.length > 0) {
      console.log('📋 First result sample:', {
        sessionId: results[0].sessionId,
        userId: results[0].userId,
        finalMmseScore: results[0].finalMmseScore,
        createdAt: results[0].createdAt
      });
    }

    return NextResponse.json({
      success: true,
      data: results,
      count: results.length,
      filters: { sessionId, userId, usageMode, limit }
    });

  } catch (error) {
    console.error('❌ Error retrieving cognitive assessment results:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}
