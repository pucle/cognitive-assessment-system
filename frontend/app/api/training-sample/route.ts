import { NextResponse } from 'next/server'
import { drizzle } from 'drizzle-orm/neon-http'
import { neon } from '@neondatabase/serverless'
import { trainingSamples } from '@/db/schema'

export async function POST(req: Request) {
  try {
    const form = await req.formData()
    const sessionId = String(form.get('sessionId') || '')
    const userEmail = String(form.get('userEmail') || '')
    const userName = String(form.get('userName') || '')
    const questionId = String(form.get('questionId') || '')
    const questionText = String(form.get('questionText') || '')
    const audioFilename = String(form.get('audioFilename') || '')
    const audioUrl = String(form.get('audioUrl') || '')
    const autoTranscript = String(form.get('autoTranscript') || '')
    const manualTranscript = String(form.get('manualTranscript') || '')

    if (!userEmail || !questionId || !questionText) {
      return NextResponse.json({ success: false, error: 'Missing required fields' }, { status: 400 })
    }

    const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '')
    const db = drizzle(sql)

    await db.insert(trainingSamples).values({
      sessionId: Number.isNaN(parseInt(sessionId, 10)) ? 0 : parseInt(sessionId, 10),
      userEmail,
      userName,
      questionId: Number.isNaN(parseInt(questionId, 10)) ? 0 : parseInt(questionId, 10),
      questionText,
      audioFilename,
      audioUrl,
      autoTranscript,
      manualTranscript,
    })

    return NextResponse.json({ success: true })
  } catch (e: any) {
    console.error('training-sample error', e)
    return NextResponse.json({ success: false, error: e?.message || 'Internal error' }, { status: 500 })
  }
}


