import { NextRequest, NextResponse } from 'next/server';
import { neon } from '@neondatabase/serverless';
import { cognitiveAssessmentResults } from '@/db/schema';
import { sql, desc, eq } from 'drizzle-orm';
import { drizzle } from 'drizzle-orm/neon-http';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const email = searchParams.get('email');
    const usageMode = searchParams.get('usageMode') || 'community';
    const limit = parseInt(searchParams.get('limit') || '50');

    if (!email) {
      return NextResponse.json({
        success: false,
        error: 'Email is required for community stats'
      }, { status: 400 });
    }

    const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

    console.log(`🔍 Fetching community stats for email: ${email}`);

    if (databaseUrl && !databaseUrl.startsWith('file:')) {
      // PostgreSQL database
      const sqlConnection = neon(databaseUrl);
      const db = drizzle(sqlConnection);

      // Query for community assessments with userId pattern: community-assessments-[index]
      // and matching email in userInfo
      const results = await db
        .select()
        .from(cognitiveAssessmentResults)
        .where(sql`${cognitiveAssessmentResults.usageMode} = ${usageMode}`)
        .orderBy(desc(cognitiveAssessmentResults.createdAt))
        .limit(limit);

      // Filter results by email in userInfo JSON
      const filteredResults = results.filter(result => {
        if (result.userInfo && typeof result.userInfo === 'object') {
          const userInfo = result.userInfo as any;
          return userInfo.email === email;
        }
        return false;
      });

      console.log(`✅ Found ${filteredResults.length} community results for ${email}`);

      return NextResponse.json({
        success: true,
        count: filteredResults.length,
        data: filteredResults,
        email: email,
        mode: usageMode
      });
    } else {
      // SQLite fallback
      console.log('📱 Using SQLite database for community stats');
      const { default: Database } = await import('better-sqlite3');
      const { drizzle: drizzleSQLite } = await import('drizzle-orm/better-sqlite3');

      const sqlite = new Database(databaseUrl?.replace('file:', '') || './cognitive_assessment.db');
      const db = drizzleSQLite(sqlite);

      try {
        const results = await db
          .select()
          .from(cognitiveAssessmentResults)
          .where(eq(cognitiveAssessmentResults.usageMode, usageMode))
          .orderBy(desc(cognitiveAssessmentResults.createdAt))
          .limit(limit);

        // Filter results by email in userInfo JSON
        const filteredResults = results.filter(result => {
          if (result.userInfo && typeof result.userInfo === 'object') {
            const userInfo = result.userInfo as any;
            return userInfo.email === email;
          }
          return false;
        });

        console.log(`✅ Found ${filteredResults.length} community results for ${email}`);

        return NextResponse.json({
          success: true,
          count: filteredResults.length,
          data: filteredResults,
          email: email,
          mode: usageMode
        });
      } finally {
        sqlite.close();
      }
    }

  } catch (error) {
    console.error('❌ Error fetching community stats:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

