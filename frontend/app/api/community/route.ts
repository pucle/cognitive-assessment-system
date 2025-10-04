import { NextResponse, NextRequest } from 'next/server'
import db from '@/db/drizzle'
import { communityAssessments } from '@/db/schema'
import { desc, sql } from 'drizzle-orm'

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    // Add pagination support
    const { searchParams } = new URL(request.url);
    const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 100); // Max 100 records
    const offset = parseInt(searchParams.get('offset') || '0');

    // Use optimized query with LIMIT/OFFSET
    const results = await db
      .select()
      .from(communityAssessments)
      .orderBy(desc(communityAssessments.createdAt))
      .limit(limit)
      .offset(offset);

    const responseTime = Date.now() - startTime;

    return NextResponse.json({
      success: true,
      message: 'Community assessments retrieved successfully',
      results: results,
      pagination: {
        limit,
        offset,
        count: results.length
      },
      performance: {
        responseTime: `${responseTime}ms`
      }
    }, {
      headers: {
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600' // Cache for 5 minutes
      }
    });

  } catch (e: any) {
    const responseTime = Date.now() - startTime;
    console.error('community API fetch error:', e);
    return NextResponse.json({
      success: false,
      error: e?.message || 'Internal error',
      results: [],
      performance: {
        responseTime: `${responseTime}ms`,
        error: true
      }
    }, { status: 500 });
  }
}
