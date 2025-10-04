import { NextResponse } from 'next/server'
import { drizzle } from 'drizzle-orm/neon-http'
import { neon } from '@neondatabase/serverless'
import { communityAssessments } from '@/db/schema'
import { eq } from 'drizzle-orm'

export async function POST(req: Request) {
  try {
    const form = await req.formData()
    const sessionId = String(form.get('sessionId') || '')
    const finalMmse = Number(form.get('finalMmse') || 0)
    const overallGptScore = Number(form.get('overallGptScore') || 0)
    const resultsJson = String(form.get('resultsJson') || '')

    if (!sessionId) {
      return NextResponse.json({ success: false, error: 'Missing sessionId' }, { status: 400 })
    }

    const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '')
    const db = drizzle(sql)

    await db.update(communityAssessments)
      .set({ finalMmse, overallGptScore, resultsJson, status: 'completed' })
      .where(eq(communityAssessments.sessionId, sessionId))

    return NextResponse.json({ success: true })
  } catch (e: any) {
    console.error('community finalize error', e)
    return NextResponse.json({ success: false, error: e?.message || 'Internal error' }, { status: 500 })
  }
}


