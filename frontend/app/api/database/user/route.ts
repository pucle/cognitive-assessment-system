import { NextRequest, NextResponse } from 'next/server';
import db from '@/db/drizzle';
import { users } from '@/db/schema';
import { eq } from 'drizzle-orm';

// Interface cho user data
interface UserData {
  id: string;
  name: string;
  age: string;
  gender: string;
  email: string;
  phone: string | null;
  avatar?: string | null;
  created_at?: string;
  updated_at?: string;
}

// Database connection is now imported from @/db/drizzle

// Lấy user từ database
async function getUserFromDatabase(userId?: string, email?: string): Promise<UserData | null> {
  try {
    let user;

    if (userId) {
      // Tìm theo ID
      const result = await db.select().from(users).where(eq(users.id, parseInt(userId)));
      user = result[0];
    } else if (email) {
      // Tìm theo email
      const result = await db.select().from(users).where(eq(users.email, email));
      user = result[0];
    }

    if (user) {
      return {
        id: user.id.toString(),
        name: user.name || user.displayName || '',
        age: user.age || '',
        gender: user.gender || '',
        email: user.email || '',
        phone: user.phone || '',
        avatar: user.avatar || user.imageSrc || null,
        created_at: user.createdAt?.toISOString(),
        updated_at: user.updatedAt?.toISOString()
      };
    }

    return null;
  } catch (error) {
    console.error('Error getting user from database:', error);
    // Return null instead of throwing to allow fallback to demo data
    return null;
  }
}

// GET endpoint để lấy user data
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');
    const email = searchParams.get('email');

    if (!userId && !email) {
      return NextResponse.json({
        success: false,
        error: 'Missing userId or email parameter'
      }, { status: 400 });
    }

    const user = await getUserFromDatabase(userId || undefined, email || undefined);

    if (user) {
      // User found in database
      return NextResponse.json({
        success: true,
        user: user,
        source: 'database'
      });
    } else {
      // User not found, return demo data
      console.log('User not found in database, returning demo data');
      const demoUser = {
        id: userId || '1',
        name: 'Lê Đình Phúc',
        age: '17',
        gender: 'Nam',
        email: email || 'ledinhphuc1408@gmail.com',
        phone: '0934865593',
        avatar: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };

      return NextResponse.json({
        success: true,
        user: demoUser,
        source: 'demo',
        message: 'User not found in database, showing demo data'
      });
    }

  } catch (error) {
    console.error('Database API GET error:', error);
    return NextResponse.json({
      success: false,
      error: 'Internal server error'
    }, { status: 500 });
  }
}

// POST endpoint để cập nhật user data
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, userId, email, ...userData } = body;

    if (action === 'get_user') {
      const user = await getUserFromDatabase(userId || undefined, email || undefined);

      if (user) {
        // User found in database
        return NextResponse.json({
          success: true,
          user: user,
          source: 'database'
        });
      } else {
        // User not found, return demo data
        console.log('User not found in database, returning demo data');
        const demoUser = {
          id: userId || '1',
          name: 'Lê Đình Phúc',
          age: '25',
          gender: 'Nam',
          email: email || 'ledinhphuc1408@gmail.com',
          phone: '0123456789',
          avatar: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        };

        return NextResponse.json({
          success: true,
          user: demoUser,
          source: 'demo',
          message: 'User not found in database, showing demo data'
        });
      }
    }

    return NextResponse.json({
      success: false,
      error: 'Invalid action'
    }, { status: 400 });

  } catch (error) {
    console.error('Database API POST error:', error);
    return NextResponse.json({
      success: false,
      error: 'Internal server error'
    }, { status: 500 });
  }
}
