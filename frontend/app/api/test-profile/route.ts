// Test API route để kiểm tra profile functionality mà không cần authentication
import { NextResponse } from "next/server";
import db from "@/db/drizzle";
import { users } from "@/db/schema";
import { eq } from "drizzle-orm";

export async function GET() {
  try {
    console.log("🧪 Test GET: Fetching all users...");
    
    // Lấy tất cả users để test
    const allUsers = await db.select().from(users);
    
    return NextResponse.json({
      success: true,
      message: "Test GET successful",
      users: allUsers,
      count: allUsers.length
    });
  } catch (error) {
    console.error("Test GET Error:", error);
    return NextResponse.json({ 
      error: (error as Error).message || "Internal server error" 
    }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    console.log("🧪 Test POST: Creating test user...");
    
    const body = await request.json();
    const { name, age, gender, phone, title, imageSrc, mmseScore } = body;

    // Validate required fields
    if (!name || !age || !gender || !phone) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    // Tạo test user với email cố định
    const testEmail = "test@example.com";
    
    // Kiểm tra xem test user đã tồn tại chưa
    const existing = await db
      .select()
      .from(users)
      .where(eq(users.email, testEmail));

    if (existing.length > 0) {
      // Cập nhật test user - update individual columns
      await db
        .update(users)
        .set({
          name: name, // Update name column directly
          age: age, // Update age column directly
          gender: gender, // Update gender column directly
          phone: phone, // Update phone column directly
          title: title || existing[0].title || "", // Update title column directly
          avatar: imageSrc || existing[0].avatar || "", // Update avatar column directly
          mmseScore: mmseScore !== undefined ? String(mmseScore) : existing[0].mmseScore, // Update mmseScore column directly
        })
        .where(eq(users.email, testEmail));

      console.log("✅ Test user updated successfully");
    } else {
      // Tạo mới test user - insert individual columns
      await db.insert(users).values({
        email: testEmail,
        name: name, // Insert name column directly
        age: age, // Insert age column directly
        gender: gender, // Insert gender column directly
        phone: phone, // Insert phone column directly
        title: title || "", // Insert title column directly
        avatar: imageSrc || "", // Insert avatar column directly
        mmseScore: mmseScore !== undefined ? String(mmseScore) : null, // Insert mmseScore column directly
      });

      console.log("✅ Test user created successfully");
    }

    return NextResponse.json({ 
      success: true, 
      message: "Test user saved successfully",
      email: testEmail
    });

  } catch (error: unknown) {
    console.error("Test POST Error:", error);
    return NextResponse.json({ 
      error: (error as Error).message || "Internal server error" 
    }, { status: 500 });
  }
}
