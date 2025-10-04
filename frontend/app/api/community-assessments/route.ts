import { NextResponse } from 'next/server'
import { drizzle } from 'drizzle-orm/neon-http'
import { neon } from '@neondatabase/serverless'
import { communityAssessments } from '@/db/schema'
import { desc } from 'drizzle-orm'

export async function GET() {
  try {
    const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '')
    const db = drizzle(sql)

    // Fetch all community assessments (for healthcare providers)
    const results = await db
      .select()
      .from(communityAssessments)
      .orderBy(desc(communityAssessments.createdAt))

    return NextResponse.json({
      success: true,
      results: results
    })

  } catch (e: any) {
    console.error('community-assessments fetch error', e)
    return NextResponse.json({
      success: false,
      error: e?.message || 'Internal error',
      results: []
    }, { status: 500 })
  }
}
