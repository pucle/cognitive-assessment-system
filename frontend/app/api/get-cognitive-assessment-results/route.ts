import { NextRequest, NextResponse } from 'next/server';
import { neon } from '@neondatabase/serverless';
import { desc, eq } from 'drizzle-orm';
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
        // For SQLite, build query with proper where conditions using any type to avoid Drizzle type issues
        let query: any = db.select().from(cognitiveAssessmentResults);

        // Apply filters one by one to avoid type narrowing issues
        if (sessionId) {
          query = query.where(eq(cognitiveAssessmentResults.sessionId, sessionId));
        }
        if (userId) {
          query = query.where(eq(cognitiveAssessmentResults.userId, userId));
        }
        if (usageMode) {
          query = query.where(eq(cognitiveAssessmentResults.usageMode, usageMode));
        }

        query = query.orderBy(desc(cognitiveAssessmentResults.createdAt));
        results = await query.limit(limit);

        // Process results to parse userInfo JSON
        results = results.map((result: unknown) => {
          const data = result as { userInfo?: unknown; [key: string]: unknown };
          let userInfo = { name: 'N/A', email: 'N/A', age: 'N/A', gender: 'N/A' };

          if (data.userInfo) {
            try {
              // Parse JSON string from database
              if (typeof data.userInfo === 'string') {
                userInfo = { ...userInfo, ...JSON.parse(data.userInfo) };
              } else if (typeof data.userInfo === 'object') {
                userInfo = { ...userInfo, ...(data.userInfo as object) };
              }
            } catch (e) {
              console.warn('Failed to parse userInfo JSON:', e);
            }
          }

          return {
            ...data,
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
      const neonClient = neon(databaseUrl);
      const { drizzle } = await import('drizzle-orm/neon-http');
      const db = drizzle(neonClient);

      // For PostgreSQL, build query with proper where conditions
      let query: any = db.select().from(cognitiveAssessmentResults);

      // Apply filters one by one to avoid type narrowing issues
      if (sessionId) {
        query = query.where(eq(cognitiveAssessmentResults.sessionId, sessionId));
      }
      if (userId) {
        query = query.where(eq(cognitiveAssessmentResults.userId, userId));
      }
      if (usageMode) {
        query = query.where(eq(cognitiveAssessmentResults.usageMode, usageMode));
      }

      query = query.orderBy(desc(cognitiveAssessmentResults.createdAt));

      results = await query.limit(limit);

      // Process results to parse userInfo JSON
      results = results.map((result: unknown) => {
        const data = result as { userInfo?: unknown; [key: string]: unknown };
        let userInfo = { name: 'N/A', email: 'N/A', age: 'N/A', gender: 'N/A' };

        if (data.userInfo) {
          try {
            // Parse JSON from database
            if (typeof data.userInfo === 'string') {
              userInfo = { ...userInfo, ...JSON.parse(data.userInfo) };
            } else if (typeof data.userInfo === 'object') {
              userInfo = { ...userInfo, ...(data.userInfo as object) };
            }
          } catch (e) {
            console.warn('Failed to parse userInfo JSON:', e);
          }
        }

        return {
          ...data,
          userInfo
        };
      });

      console.log(`✅ PostgreSQL query successful, found ${results.length} results`);
    }

    // Log first result for debugging
    if (results && results.length > 0) {
      const firstResult = results[0] as {
        sessionId?: string;
        userId?: string;
        finalMmseScore?: number;
        createdAt?: string;
        [key: string]: unknown;
      };
      console.log('📋 First result sample:', {
        sessionId: firstResult.sessionId,
        userId: firstResult.userId,
        finalMmseScore: firstResult.finalMmseScore,
        createdAt: firstResult.createdAt
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
