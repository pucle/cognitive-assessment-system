// app/api/profile/user/route.ts
export const runtime = 'nodejs';
import 'server-only';

import { NextRequest, NextResponse } from 'next/server';
import db from '@/db/drizzle';
import { users } from '@/db/schema';
import { eq } from 'drizzle-orm';

// Normalize input
function normalizeString(value: unknown): string {
	if (typeof value === 'string') return value.trim();
	if (value == null) return '';
	return String(value).trim();
}

export async function GET(request: NextRequest) {
	try {
		const { searchParams } = new URL(request.url);
		const userId = normalizeString(searchParams.get('userId'));
		const email = normalizeString(searchParams.get('email'));

		if (!userId && !email) {
			return NextResponse.json({ success: false, error: 'Missing userId or email' }, { status: 400 });
		}

		let result;
		if (userId) {
			const idNum = Number(userId);
			if (Number.isNaN(idNum)) {
				return NextResponse.json({ success: false, error: 'Invalid userId' }, { status: 400 });
			}
			result = await db.select().from(users).where(eq(users.id, idNum)).limit(1);
		} else {
			result = await db.select().from(users).where(eq(users.email, email)).limit(1);
		}

		const user = result?.[0];
		if (!user) {
			return NextResponse.json({ success: false, error: 'User not found' }, { status: 404 });
		}

		return NextResponse.json({
			success: true,
			user: {
				id: String(user.id),
				name: user.name || user.displayName, // Use name column directly
				age: user.age, // Use age column directly
				gender: user.gender, // Use gender column directly
				email: user.email,
				phone: user.phone || '', // Use phone column directly
				title: user.title || '', // Use title column directly
				imageSrc: user.avatar || user.imageSrc || '', // Use avatar or imageSrc
				mmseScore: user.mmseScore || '', // Use mmseScore column directly
				created_at: user.createdAt?.toISOString(),
				updated_at: user.updatedAt?.toISOString(),
			},
		});
	} catch (error) {
		console.error('GET /api/profile/user error:', error);
		return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 });
	}
}

export async function POST(request: NextRequest) {
	try {
		const body = await request.json();
		const name = normalizeString(body?.name);
		const age = normalizeString(body?.age);
		const gender = normalizeString(body?.gender);
		const email = normalizeString(body?.email);
		const phone = normalizeString(body?.phone);
		const title = normalizeString(body?.title);
		const imageSrc = normalizeString(body?.imageSrc);
		const mmseScore = normalizeString(body?.mmseScore);

		if (!name || !age || !gender || !email) {
			return NextResponse.json({ success: false, error: 'Missing required fields' }, { status: 400 });
		}

		// Check if user exists by email
		const existing = await db.select().from(users).where(eq(users.email, email)).limit(1);

		let savedId: number;
		if (existing.length > 0) {
			const current = existing[0];
			await db
				.update(users)
				.set({
					name: name || current.name, // Update name column directly
					age: age || current.age, // Update age column directly
					gender: gender || current.gender, // Update gender column directly
					phone: phone || current.phone, // Update phone column directly
					title: title || current.title, // Update title column directly
					avatar: imageSrc || current.avatar, // Update avatar column directly
					mmseScore: mmseScore || current.mmseScore, // Update mmseScore column directly
					updatedAt: new Date(),
				})
				.where(eq(users.id, current.id));
			savedId = current.id;
		} else {
			const inserted = await db
				.insert(users)
				.values({
					email,
					name: name, // Insert name column directly
					age: age, // Insert age column directly
					gender: gender, // Insert gender column directly
					phone: phone, // Insert phone column directly
					title: title, // Insert title column directly
					avatar: imageSrc, // Insert avatar column directly
					mmseScore: mmseScore // Insert mmseScore column directly
				})
				.returning({ id: users.id });
			savedId = inserted[0].id;
		}

		return NextResponse.json({ success: true, id: String(savedId) });
	} catch (error) {
		console.error('POST /api/profile/user error:', error);
		return NextResponse.json({ success: false, error: 'Internal server error' }, { status: 500 });
	}
}
