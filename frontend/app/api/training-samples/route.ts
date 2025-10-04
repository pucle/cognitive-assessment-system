export const runtime = 'nodejs';
import 'server-only';

import { NextResponse } from 'next/server'
import { auth, currentUser } from "@clerk/nextjs/server";
import db from '@/db/drizzle'
import { trainingSamples } from '@/db/schema'
import { desc, eq } from 'drizzle-orm'

export async function GET() {
  try {
    let userId: string | null = null;
    let user: any = null;
    let email: string | undefined;

    // Check if Clerk is available and properly configured
    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

    if (isClerkAvailable) {
      try {
        const authResult = await auth();
        userId = authResult.userId;
        user = await currentUser();
        email = user?.primaryEmailAddress?.emailAddress;

        // If auth fails, fall back to demo mode
        if (!userId || !email) {
          console.warn('Training-samples API: Auth failed, falling back to demo mode');
          userId = 'demo-user';
          email = 'demo@example.com';
        }
      } catch (authError) {
        console.warn("Auth error, using demo mode:", authError.message);
        userId = 'demo-user';
        email = 'demo@example.com';
      }
    } else {
      // Demo mode - use fallback values
      console.warn('Training-samples API: Clerk not available, using demo mode');
      userId = 'demo-user';
      email = 'demo@example.com';
    }

    // Use the optimized database connection
    const startTime = Date.now();

    // Fetch training samples for the current user with optimized query
    const results = await db
      .select()
      .from(trainingSamples)
      .where(eq(trainingSamples.userEmail, email))
      .orderBy(desc(trainingSamples.createdAt))
      .limit(100); // Limit results for performance

    const responseTime = Date.now() - startTime;

    return NextResponse.json({
      success: true,
      message: 'Training samples retrieved successfully',
      results: results,
      performance: {
        responseTime: `${responseTime}ms`,
        count: results.length
      }
    }, {
      headers: {
        'Cache-Control': 'private, s-maxage=60, stale-while-revalidate=120' // Cache for 1 minute
      }
    });

  } catch (e: any) {
    console.error('training-samples fetch error', e)
    return NextResponse.json({
      success: false,
      error: e?.message || 'Internal error',
      results: []
    }, { status: 500 })
  }
}
