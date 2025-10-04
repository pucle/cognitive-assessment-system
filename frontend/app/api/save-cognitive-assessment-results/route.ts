export const runtime = 'nodejs';
import 'server-only';

import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs/server";
import { drizzle } from 'drizzle-orm/neon-http';
import { neon } from '@neondatabase/serverless';
import { cognitiveAssessmentResults } from '@/db/schema';

export async function POST(request: NextRequest) {
  try {
    let clerkUserId: string | null = null;

    // Check if Clerk is available
    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

    if (isClerkAvailable) {
      try {
        clerkUserId = (await auth()).userId;
      } catch (authError) {
        console.error("Auth error:", authError);
        return NextResponse.json({ error: "Authentication failed" }, { status: 401 });
      }
    } else {
      // Demo mode - use fallback values
      console.warn('Save-cognitive-assessment-results API: Clerk not available, using demo mode');
      clerkUserId = 'demo-user';
    }

    const body = await request.json();
    const {
      sessionId,
      userId,
      userInfo,
      startedAt,
      totalQuestions,
      answeredQuestions,
      completionRate,
      memoryScore,
      cognitiveScore,
      finalMmseScore,
      overallGptScore,
      questionResults,
      audioFiles,
      recordingsPath,
      cognitiveAnalysis,
      audioFeatures,
      usageMode,
      assessmentType
    } = body;

    // Validate required fields
    if (!sessionId) {
      return NextResponse.json({
        success: false,
        error: 'Session ID is required'
      }, { status: 400 });
    }

    const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

    // Check if database URL is configured
    if (!databaseUrl) {
      console.warn('⚠️ Database not configured, running in demo mode');
      // Return success response for demo purposes
      return NextResponse.json({
        success: true,
        message: 'Demo mode: Assessment results would be saved to database',
        id: `demo_${typeof window !== 'undefined' ? Date.now() : Math.floor(Math.random() * 1000000)}`,
        mode: 'demo'
      });
    }

    const sql = neon(databaseUrl);
    const db = drizzle(sql);

    // Insert cognitive assessment result
    const result = await db.insert(cognitiveAssessmentResults).values({
      sessionId,
      userId: userId || clerkUserId || null,
      userInfo,
      startedAt: startedAt ? new Date(startedAt) : null,
      totalQuestions,
      answeredQuestions,
      completionRate,
      memoryScore,
      cognitiveScore,
      finalMmseScore,
      overallGptScore,
      questionResults,
      audioFiles,
      recordingsPath,
      cognitiveAnalysis,
      audioFeatures,
      usageMode: usageMode || 'personal',
      assessmentType: assessmentType || 'cognitive',
      status: 'completed'
    }).returning({ id: cognitiveAssessmentResults.id });

    console.log('✅ Cognitive assessment result saved to database:', {
      id: result[0].id,
      sessionId,
      userId: userId || clerkUserId
    });

    return NextResponse.json({
      success: true,
      id: result[0].id,
      message: 'Cognitive assessment result saved successfully'
    });

  } catch (error) {
    console.error('❌ Error saving cognitive assessment result:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  // Import training samples data to cognitive_assessment_results
  try {
    const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || 'file:./cognitive_assessment.db';

    let db: any;
    let sqlite: any = null;
    if (databaseUrl.startsWith('file:')) {
      const { default: Database } = await import('better-sqlite3');
      const { drizzle: drizzleSQLite } = await import('drizzle-orm/better-sqlite3');
      sqlite = new Database(databaseUrl.replace('file:', ''));
      db = drizzleSQLite(sqlite);
    } else {
      const sql = neon(databaseUrl);
      db = drizzle(sql);
    }

    // Import from training_samples table
    if (databaseUrl.startsWith('file:') && sqlite) {
      const trainingSamples = sqlite.prepare('SELECT * FROM training_samples ORDER BY created_at DESC').all();

      if (trainingSamples.length > 0) {
        const insert = sqlite.prepare(`
          INSERT OR IGNORE INTO cognitive_assessment_results (
            sessionId, userId, userInfo, finalMmseScore, overallGptScore,
            questionResults, status, usageMode, assessmentType, createdAt
          ) VALUES (
            @sessionId, @userId, @userInfo, @finalMmseScore, @overallGptScore,
            @questionResults, @status, @usageMode, @assessmentType, @createdAt
          )
        `);

        sqlite.transaction((samples: any[]) => {
          samples.forEach(sample => {
            const sessionId = `training_${sample.id}_${Date.now()}`;
            insert.run({
              sessionId,
              userId: sample.user_email || 'imported_user',
              userInfo: JSON.stringify({
                name: sample.user_name || 'Imported User',
                email: sample.user_email || ''
              }),
              finalMmseScore: sample.auto_transcript ? parseInt(sample.auto_transcript.split(':')[1]?.trim()) : null,
              overallGptScore: sample.manual_transcript ? parseFloat(sample.manual_transcript.split(':')[1]?.trim()) : null,
              questionResults: JSON.stringify({
                training_sample_id: sample.id,
                question: sample.question_text,
                auto_transcript: sample.auto_transcript,
                manual_transcript: sample.manual_transcript
              }),
              status: 'completed',
              usageMode: 'personal',
              assessmentType: 'imported',
              createdAt: sample.created_at || new Date().toISOString()
            });
          });
        })(trainingSamples);

        return NextResponse.json({
          success: true,
          message: `Imported ${trainingSamples.length} training samples`,
          imported: trainingSamples.length
        });
      }
    }

    return NextResponse.json({
      success: false,
      error: 'No training samples found or import not supported'
    });

  } catch (error) {
    console.error('Error importing training samples:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const sessionId = searchParams.get('sessionId');
    let userId = searchParams.get('userId');

    // Try to infer userId from Clerk if not provided
    try {
      if (!userId) {
        const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
        const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');
        if (isClerkAvailable) {
          const authMod = await import('@clerk/nextjs/server');
          const clerkUserId = (await authMod.auth()).userId;
          if (clerkUserId) userId = clerkUserId;
        }
      }
    } catch {}

    // If neither is provided even after Clerk inference, we will return latest results instead of 400

    const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || 'file:./cognitive_assessment.db';

    let db: any;
    let sqlite: any = null;
    if (databaseUrl.startsWith('file:')) {
      // SQLite database
      const { default: Database } = await import('better-sqlite3');
      const { drizzle: drizzleSQLite } = await import('drizzle-orm/better-sqlite3');
      sqlite = new Database(databaseUrl.replace('file:', ''));

      // Initialize tables if they don't exist
      try {
        sqlite.exec(`
          CREATE TABLE IF NOT EXISTS cognitive_assessment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sessionId TEXT NOT NULL,
            userId TEXT,
            userInfo TEXT,
            startedAt TEXT,
            completedAt TEXT,
            totalQuestions INTEGER DEFAULT 0,
            answeredQuestions INTEGER DEFAULT 0,
            completionRate REAL,
            memoryScore REAL,
            cognitiveScore REAL,
            finalMmseScore INTEGER,
            overallGptScore REAL,
            questionResults TEXT,
            audioFiles TEXT,
            recordingsPath TEXT,
            cognitiveAnalysis TEXT,
            audioFeatures TEXT,
            status TEXT DEFAULT 'completed',
            usageMode TEXT DEFAULT 'personal',
            assessmentType TEXT DEFAULT 'cognitive',
            createdAt TEXT DEFAULT '',
            updatedAt TEXT DEFAULT ''
          );
        `);
        console.log('SQLite tables initialized successfully');
      } catch (initError) {
        console.error('Failed to initialize SQLite tables:', initError);
      }

      db = drizzleSQLite(sqlite);
    } else {
      // PostgreSQL database
      const sql = neon(databaseUrl);
      db = drizzle(sql);
    }

    // Import eq and and for filtering
    const { eq, and, desc } = await import('drizzle-orm');

    // Build the where conditions
    const conditions = [] as any[];
    if (sessionId) {
      conditions.push(eq(cognitiveAssessmentResults.sessionId, sessionId));
    }
    if (userId) {
      conditions.push(eq(cognitiveAssessmentResults.userId, userId));
    }

    // Build and execute the query with proper conditions
    let results: any[];

    if (databaseUrl.startsWith('file:') && sqlite) {
      // SQLite path: use raw SQL to avoid dialect mismatches
      try {
        if (conditions.length === 0) {
          // No filters provided: return latest 1-10 rows instead of 400 to support menu fallback
          results = sqlite.prepare(`
            SELECT * FROM cognitive_assessment_results
            ORDER BY createdAt DESC
            LIMIT 10
          `).all();
        } else if (conditions.length === 1) {
          if (sessionId) {
            results = sqlite.prepare(`
              SELECT * FROM cognitive_assessment_results
              WHERE sessionId = ?
              ORDER BY createdAt DESC
              LIMIT 10
            `).all(sessionId);
          } else if (userId) {
            results = sqlite.prepare(`
              SELECT * FROM cognitive_assessment_results
              WHERE userId = ?
              ORDER BY createdAt DESC
              LIMIT 10
            `).all(userId);
          } else {
            results = [];
          }
        } else {
          // Multiple conditions - use AND
          results = sqlite.prepare(`
            SELECT * FROM cognitive_assessment_results
            WHERE sessionId = ? AND userId = ?
            ORDER BY createdAt DESC
            LIMIT 10
          `).all(sessionId, userId);
        }
      } catch (sqliteError) {
        console.error('SQLite query error:', sqliteError);
        results = [];
      }
    } else {
      // PostgreSQL path
      if (conditions.length === 0) {
        // No filters provided: return latest 1-10 rows instead of 400 to support menu fallback
        results = await db.select()
          .from(cognitiveAssessmentResults)
          .limit(10)
          .orderBy(desc(cognitiveAssessmentResults.createdAt));
      } else if (conditions.length === 1) {
        results = await db.select()
          .from(cognitiveAssessmentResults)
          .where(conditions[0])
          .limit(10)
          .orderBy(desc(cognitiveAssessmentResults.createdAt));
      } else {
        results = await db.select()
          .from(cognitiveAssessmentResults)
          .where(and(...conditions))
          .limit(10)
          .orderBy(desc(cognitiveAssessmentResults.createdAt));
      }
    }

    // DISABLED: Add test data if no results found - THIS CODE IS DISABLED AND SHOULD NOT RUN
    /*
    if (results.length === 0) {
      console.log('No database results found, adding test data...');
      const testData = [
        {
          sessionId: 'test_session_001',
          userId: 'demo_user',
          userInfo: { name: 'Test User 1', age: '25', gender: 'male', email: 'test1@example.com' },
          startedAt: new Date(Date.now() - 3600000),
          totalQuestions: 12,
          answeredQuestions: 12,
          completionRate: 100,
          finalMmseScore: 28,
          overallGptScore: 8.5,
          questionResults: [{ questionId: 'Q1', score: 1 }, { questionId: 'Q2', score: 1 }],
          status: 'completed',
          usageMode: 'personal'
        },
        {
          sessionId: 'test_session_002',
          userId: 'demo_user',
          userInfo: { name: 'Test User 2', age: '30', gender: 'female', email: 'test2@example.com' },
          startedAt: new Date(Date.now() - 7200000),
          totalQuestions: 12,
          answeredQuestions: 12,
          completionRate: 100,
          finalMmseScore: 25,
          overallGptScore: 7.8,
          questionResults: [{ questionId: 'Q1', score: 1 }, { questionId: 'Q2', score: 1 }],
          status: 'completed',
          usageMode: 'personal'
        }
      ];

      try {
        if (databaseUrl.startsWith('file:') && sqlite) {
          // SQLite insert
          const insert = sqlite.prepare(`
            INSERT INTO cognitive_assessment_results (
              sessionId, userId, userInfo, startedAt, totalQuestions,
              answeredQuestions, completionRate, finalMmseScore, overallGptScore,
              questionResults, status, usageMode, assessmentType, createdAt, updatedAt
            ) VALUES (
              @sessionId, @userId, @userInfo, @startedAt, @totalQuestions,
              @answeredQuestions, @completionRate, @finalMmseScore, @overallGptScore,
              @questionResults, @status, @usageMode, @assessmentType, @createdAt, @updatedAt
            )
          `);

          sqlite.transaction((rows: any[]) => {
            rows.forEach(row => insert.run({
              sessionId: row.sessionId,
              userId: row.userId,
              userInfo: JSON.stringify(row.userInfo || {}),
              startedAt: row.startedAt ? row.startedAt.toISOString() : new Date().toISOString(),
              totalQuestions: row.totalQuestions,
              answeredQuestions: row.answeredQuestions,
              completionRate: row.completionRate,
              finalMmseScore: row.finalMmseScore,
              overallGptScore: row.overallGptScore,
              questionResults: JSON.stringify(row.questionResults || []),
              status: row.status || 'completed',
              usageMode: row.usageMode || 'personal',
              assessmentType: row.assessmentType || 'cognitive',
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            }));
          })(testData as any[]);

          // Re-fetch after insert
          results = sqlite.prepare(`
            SELECT * FROM cognitive_assessment_results
            ORDER BY createdAt DESC
            LIMIT 10
          `).all();
        } else {
          // PostgreSQL insert
          for (const data of testData) {
            await db.insert(cognitiveAssessmentResults).values(data);
          }

          // Re-fetch with test data
          results = await db.select()
            .from(cognitiveAssessmentResults)
            .limit(10)
            .orderBy(desc(cognitiveAssessmentResults.createdAt));
        }

        console.log('Test data inserted successfully');

      } catch (insertError) {
        console.error('Failed to insert test data:', insertError);
      }
    } // DISABLED test data insertion
    */

    console.log('API returning results:', {
      count: results.length,
      firstResult: results[0] || 'none'
    });

    return NextResponse.json({
      success: true,
      results,
      count: results.length
    });

  } catch (error) {
    console.error('❌ Error retrieving cognitive assessment results:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}
