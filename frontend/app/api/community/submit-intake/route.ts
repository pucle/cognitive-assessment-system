import { NextResponse } from 'next/server'
import { drizzle } from 'drizzle-orm/neon-http'
import { neon } from '@neondatabase/serverless'
import { communityAssessments } from '@/db/schema'

export async function POST(req: Request) {
  try {
    const form = await req.formData()
    const sessionId = String(form.get('sessionId') || '')
    const name = String(form.get('name') || '')
    const email = String(form.get('email') || '')
    const age = String(form.get('age') || '')
    const gender = String(form.get('gender') || '')
    const phone = String(form.get('phone') || '')

    if (!email || !sessionId) {
      return NextResponse.json({ success: false, error: 'Missing email or sessionId' }, { status: 400 })
    }

    const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '')
    const db = drizzle(sql)

    await db.insert(communityAssessments).values({
      sessionId,
      name,
      email,
      age,
      gender,
      phone,
      status: 'pending'
    })

    return NextResponse.json({ success: true })
  } catch (e: any) {
    console.error('community submit-intake error', e)
    return NextResponse.json({ success: false, error: e?.message || 'Internal error' }, { status: 500 })
  }
}


