import { NextResponse } from 'next/server'
import nodemailer from 'nodemailer'

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
})

export async function POST(req: Request) {
  try {
    const form = await req.formData()
    const email = String(form.get('email') || '')
    const name = String(form.get('name') || '')
    const sessionId = String(form.get('sessionId') || '')
    const finalMmse = String(form.get('finalMmse') || '0')
    const overallGptScore = String(form.get('overallGptScore') || '0')
    const summary = String(form.get('summary') || '')

    if (!email) return NextResponse.json({ success: false, error: 'Missing email' }, { status: 400 })

    const html = `
      <div style="font-family: Arial, sans-serif; max-width:600px; margin:0 auto;">
        <h2>🐠 Cá Vàng - Kết quả đánh giá</h2>
        <p>Xin chào ${name || 'bạn'},</p>
        <p>Phiên: <b>${sessionId}</b></p>
        <ul>
          <li>Điểm MMSE cuối: <b>${finalMmse}/30</b></li>
          <li>Điểm AI tổng thể: <b>${overallGptScore}/10</b></li>
        </ul>
        <pre style="background:#f8f8f8; padding:12px; border:1px solid #eee; border-radius:8px; white-space: pre-wrap;">${summary}</pre>
        <p style="color:#666; font-size:12px">Email tự động từ hệ thống Cá Vàng.</p>
      </div>
    `

    await transporter.sendMail({
      from: {
        name: 'Cá Vàng',
        address: process.env.EMAIL_USER || 'noreply@cavang.com',
      },
      to: email,
      subject: '🐠 Kết quả đánh giá nhận thức - Cá Vàng',
      html,
    })

    return NextResponse.json({ success: true })
  } catch (e: any) {
    console.error('send-result error', e)
    return NextResponse.json({ success: false, error: e?.message || 'Internal error' }, { status: 500 })
  }
}


