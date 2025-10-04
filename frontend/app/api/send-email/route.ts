// app/api/send-email/route.ts
export const runtime = 'nodejs';
import 'server-only';

import { NextResponse } from "next/server";
import { auth, currentUser } from "@clerk/nextjs/server";
import nodemailer from 'nodemailer';

// Cấu hình email transporter (sử dụng Gmail làm ví dụ)
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER, // email của bạn
    pass: process.env.EMAIL_PASS, // app password hoặc password email
  },
});

// Hoặc sử dụng SMTP custom
// const transporter = nodemailer.createTransport({
//   host: process.env.SMTP_HOST,
//   port: parseInt(process.env.SMTP_PORT || '587'),
//   secure: false,
//   auth: {
//     user: process.env.SMTP_USER,
//     pass: process.env.SMTP_PASS,
//   },
// });

export async function POST(request: Request) {
  try {
    let userId: string | null = null;
    let user: any = null;
    let userEmail: string | undefined;

    // Check if Clerk is available
    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

    if (isClerkAvailable) {
      try {
        const authResult = await auth();
        userId = authResult.userId;
        user = await currentUser();
        userEmail = user?.primaryEmailAddress?.emailAddress;
      } catch (authError) {
        console.error("Auth error:", authError);
        return NextResponse.json({ error: "Authentication failed" }, { status: 401 });
      }

      if (!userId) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }

      if (!userEmail) {
        return NextResponse.json({ error: "Email not found" }, { status: 400 });
      }
    } else {
      // Demo mode - use fallback values
      console.warn('Send-email API: Clerk not available, using demo mode');
      userId = 'demo-user';
      userEmail = 'demo@example.com';
    }

    const body = await request.json();
    const { userData } = body;

    // Template email HTML
    const emailHTML = `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            .email-container { 
                max-width: 600px; 
                margin: 0 auto; 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
            }
            .email-content { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .header { 
                text-align: center; 
                color: #333; 
                margin-bottom: 30px;
            }
            .info-row { 
                margin: 15px 0; 
                padding: 10px; 
                background: #f8f9fa; 
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }
            .label { 
                font-weight: bold; 
                color: #555; 
                display: inline-block;
                width: 120px;
            }
            .value { 
                color: #333; 
            }
            .footer { 
                text-align: center; 
                margin-top: 30px; 
                color: #666; 
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="email-content">
                <div class="header">
                    <h1>🐠 Cá Vàng - Thông tin hồ sơ của bạn</h1>
                    <p>Cảm ơn bạn đã cập nhật thông tin cá nhân!</p>
                </div>
                
                <div class="info-row">
                    <span class="label">👤 Họ và tên:</span>
                    <span class="value">${userData.name}</span>
                </div>
                
                <div class="info-row">
                    <span class="label">🎂 Tuổi:</span>
                    <span class="value">${userData.age} tuổi</span>
                </div>
                
                <div class="info-row">
                    <span class="label">👥 Giới tính:</span>
                    <span class="value">${userData.gender}</span>
                </div>
                
                <div class="info-row">
                    <span class="label">📧 Email:</span>
                    <span class="value">${userData.email}</span>
                </div>
                
                <div class="info-row">
                    <span class="label">📱 Số điện thoại:</span>
                    <span class="value">${userData.phone}</span>
                </div>
                
                ${userData.title ? `
                <div class="info-row">
                    <span class="label">🏷️ Chức danh:</span>
                    <span class="value">${userData.title}</span>
                </div>
                ` : ''}
                
                <div class="footer">
                    <p>Đây là email tự động từ hệ thống Cá Vàng.</p>
                    <p>Nếu có thắc mắc, vui lòng liên hệ với chúng tôi.</p>
                    <hr style="margin: 20px 0; border: none; border-top: 1px solid #eee;">
                    <p><small>© 2025 Cá Vàng - Ứng dụng nhỏ, ký ức lớn</small></p>
                </div>
            </div>
        </div>
    </body>
    </html>
    `;

    // Cấu hình email
    const mailOptions = {
      from: {
        name: 'Cá Vàng',
        address: process.env.EMAIL_USER || 'noreply@cavang.com'
      },
      to: userEmail,
      subject: '🐠 Cá Vàng - Xác nhận cập nhật thông tin hồ sơ',
      html: emailHTML,
      // Text version cho email clients không hỗ trợ HTML
      text: `
Cá Vàng - Thông tin hồ sơ của bạn

Cảm ơn bạn đã cập nhật thông tin cá nhân!

Thông tin của bạn:
- Họ và tên: ${userData.name}
- Tuổi: ${userData.age} tuổi  
- Giới tính: ${userData.gender}
- Email: ${userData.email}
- Số điện thoại: ${userData.phone}
${userData.title ? `- Chức danh: ${userData.title}` : ''}

Đây là email tự động từ hệ thống Cá Vàng.
Nếu có thắc mắc, vui lòng liên hệ với chúng tôi.

© 2025 Cá Vàng - Ứng dụng nhỏ, ký ức lớn
      `
    };

    // Gửi email
    const info = await transporter.sendMail(mailOptions);

    return NextResponse.json({ 
      success: true, 
      message: "Email sent successfully",
      messageId: info.messageId
    });

  } catch (error: unknown) {
    console.error("Error sending email:", error);
    let errorMessage = "Failed to send email";
    if (error instanceof Error) {
      errorMessage = error.message;
    }
    return NextResponse.json({ 
      success: false, 
      error: errorMessage
    }, { status: 500 });
  }
}   