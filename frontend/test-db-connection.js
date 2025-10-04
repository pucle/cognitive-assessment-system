// Test database connection
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function testConnection() {
  try {
    console.log('🔍 Testing database connection...');
    const result = await sql`SELECT * FROM users LIMIT 1`;
    console.log('✅ Database connection successful');
    if (result.length > 0) {
      console.log('Users table columns:', Object.keys(result[0]));
    } else {
      console.log('Users table is empty');
    }

    // Check if displayName column exists
    console.log('\n🔍 Checking for displayName column...');
    const columns = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'users' ORDER BY column_name`;
    const columnNames = columns.map(row => row.column_name);
    console.log('All columns:', columnNames.join(', '));

    if (columnNames.includes('displayname')) {
      console.log('✅ displayName column exists');
    } else {
      console.log('❌ displayName column is missing - migration may not have run');
    }

    // Also check camelCase
    if (columnNames.includes('displayName')) {
      console.log('✅ displayName column exists (camelCase)');
    }
  } catch (error) {
    console.log('❌ Database connection failed:', error.message);
  }
}

testConnection();
