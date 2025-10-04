// app/api/profile/route.ts
export const runtime = 'nodejs';
import 'server-only';
import { NextRequest, NextResponse } from "next/server";
import db from "@/db/drizzle";
import { users } from "@/db/schema";
import { eq } from "drizzle-orm";

export async function GET(request: NextRequest) {
  try {
    // Get email from query params for user lookup
    const { searchParams } = new URL(request.url);
    const email = searchParams.get('email');

    if (!email) {
      return NextResponse.json({
        success: false,
        error: "Email parameter required"
      }, { status: 400 });
    }

    // For demo purposes, return mock data if no database connection
    // In production, this would connect to actual database
    try {
      // Try to fetch from database first
      const userRecords = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);

      if (userRecords.length > 0) {
        const user = userRecords[0];
        return NextResponse.json({
          success: true,
          message: "Profile fetched successfully",
          data: {
            name: user.name || user.displayName, // Use name column directly
            age: user.age, // Use age column directly
            gender: user.gender, // Use gender column directly
            email: user.email,
            phone: user.phone, // Use phone column directly
            title: user.title, // Use title column directly
            imageSrc: user.avatar || user.imageSrc, // Use avatar or imageSrc
            mmseScore: user.mmseScore // Use mmseScore column directly
          }
        }, { status: 200 });
      }
    } catch (dbError) {
      console.warn("Database not available, using mock data:", dbError);
    }

    // Return mock data for demo
    return NextResponse.json({
      success: true,
      message: "Profile fetched successfully (demo data)",
      data: {
        name: email === "ledinhphuc1408@gmail.com" ? "Lê Đình Phúc" : "",
        age: email === "ledinhphuc1408@gmail.com" ? "25" : "",
        gender: email === "ledinhphuc1408@gmail.com" ? "Nam" : "",
        email: email,
        phone: email === "ledinhphuc1408@gmail.com" ? "0123456789" : "",
        title: "",
        imageSrc: "",
        mmseScore: null
      }
    }, { status: 200 });

  } catch (error) {
    console.error("Error fetching profile:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, name, age, gender, phone, title, imageSrc, mmseScore } = body;

    if (!email || !name || !age || !gender) {
      return NextResponse.json({
        success: false,
        error: "Required fields: email, name, age, gender"
      }, { status: 400 });
    }

    // For demo purposes, simulate database operations
    // In production, this would actually save to database
    try {
      // Try database operations first
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);

      let result;
      if (existingUser.length > 0) {
        // Update existing user - update individual columns instead of profile
        result = await db
          .update(users)
          .set({
            name: name, // Update name column
            age: age, // Update age column
            gender: gender, // Update gender column
            phone: phone, // Update phone column
            title: title, // Update title column
            avatar: imageSrc, // Update avatar column
            mmseScore: mmseScore?.toString(), // Update mmseScore column
            updatedAt: new Date()
          })
          .where(eq(users.email, email))
          .returning();
      } else {
        // Create new user - insert individual columns instead of profile
        result = await db
          .insert(users)
          .values({
            email,
            name: name, // Insert name column
            age: age, // Insert age column
            gender: gender, // Insert gender column
            phone: phone, // Insert phone column
            title: title, // Insert title column
            avatar: imageSrc, // Insert avatar column
            mmseScore: mmseScore?.toString() // Insert mmseScore column
          })
          .returning();
      }

      return NextResponse.json({
        success: true,
        message: existingUser.length > 0 ? "Profile updated successfully" : "Profile created successfully",
        data: result[0]
      }, { status: 200 });

    } catch (dbError) {
      console.warn("Database not available, simulating success:", dbError);

      // Simulate successful operation for demo
      return NextResponse.json({
        success: true,
        message: "Profile saved successfully (demo mode)",
        data: {
          id: Date.now(),
          email,
          name,
          age,
          gender,
          phone,
          title,
          imageSrc,
          mmseScore: mmseScore?.toString()
        }
      }, { status: 200 });
    }

  } catch (error: unknown) {
    console.error("Error updating profile:", error);
    return NextResponse.json({
      success: false,
      error: (error as Error).message || "Internal server error"
    }, { status: 500 });
  }
}

