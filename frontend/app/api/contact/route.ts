import { NextRequest, NextResponse } from "next/server";
import db from "@/db/drizzle";
import { contactMessages } from "@/db/schema";
import { sql } from "drizzle-orm";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, email, subject, category, message } = body || {};

    if (!name || !email || !message) {
      return NextResponse.json({ success: false, error: "Missing required fields" }, { status: 400 });
    }

    const inserted = await db.insert(contactMessages).values({
      name,
      email,
      subject: subject || null,
      category: category || null,
      message,
    }).returning({ id: contactMessages.id });

    return NextResponse.json({ success: true, id: inserted?.[0]?.id });
  } catch (error: any) {
    // Auto-create table if it doesn't exist (first run without migrations)
    const errMsg = String(error?.message || "");
    const code = (error && (error as any).code) || "";
    if (code === "42P01" || errMsg.includes("relation \"contact_messages\" does not exist")) {
      try {
        await db.execute(sql.raw(`
          CREATE TABLE IF NOT EXISTS "contact_messages" (
            "id" serial PRIMARY KEY NOT NULL,
            "name" text NOT NULL,
            "email" text NOT NULL,
            "subject" text,
            "category" text,
            "message" text NOT NULL,
            "created_at" timestamp with time zone DEFAULT now()
          );
        `));

        const body = await req.json();
        const { name, email, subject, category, message } = body || {};
        const inserted = await db.insert(contactMessages).values({
          name,
          email,
          subject: subject || null,
          category: category || null,
          message,
        }).returning({ id: contactMessages.id });
        return NextResponse.json({ success: true, id: inserted?.[0]?.id });
      } catch (createErr) {
        console.error("/api/contact autocreate error", createErr);
      }
    }

    console.error("/api/contact error", error);
    return NextResponse.json({ success: false, error: "Internal server error" }, { status: 500 });
  }
}


