export const runtime = 'nodejs';
import 'server-only';

import { NextRequest, NextResponse } from 'next/server';
import { drizzle } from 'drizzle-orm/neon-http';
import { neon } from '@neondatabase/serverless';
import { cognitiveAssessmentResults } from '@/db/schema';
import { and, desc, eq } from 'drizzle-orm';

function pickNumber(...vals: any[]): number | null {
	for (const v of vals) {
		if (typeof v === 'number' && !Number.isNaN(v)) return v;
		if (v && typeof v === 'object') {
			// try nested combined_score
			if (typeof (v as any).combined_score === 'number') return (v as any).combined_score;
		}
	}
	return null;
}

function normalizeSessionId(item: any, index: number): string {
	const sid = item?.sessionId || item?.session_id || item?.session || null;
	if (sid && typeof sid === 'string') return sid;
	const ts = item?.timestamp || (typeof window !== 'undefined' ? Date.now() : Math.floor(Math.random() * 1000000));
	return `py_${ts}_${index}`;
}

export async function POST(_req: NextRequest) {
	try {
		// Temporarily disabled - backend doesn't have /results endpoint
		return NextResponse.json({
			success: true,
			synced: 0,
			skipped: 0,
			message: 'Sync endpoint temporarily disabled - no backend /results endpoint available'
		});
	} catch (e: any) {
		console.error('sync-python-results error', e);
		return NextResponse.json({ success: false, error: e?.message || 'Internal error' }, { status: 500 });
	}
}

export async function GET(req: NextRequest) {
	// Allow GET for convenience
	return POST(req);
}
