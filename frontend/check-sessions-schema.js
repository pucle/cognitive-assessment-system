// Check current sessions table schema
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function checkSessionsSchema() {
  try {
    console.log('🔍 Checking sessions table schema...');

    // Get all columns in sessions table
    const columns = await sql`SELECT column_name, data_type, is_nullable, column_default
                             FROM information_schema.columns
                             WHERE table_name = 'sessions'
                             ORDER BY ordinal_position`;

    console.log('\n📋 Current sessions table columns:');
    columns.forEach(col => {
      console.log(`  - ${col.column_name}: ${col.data_type}${col.is_nullable === 'NO' ? ' NOT NULL' : ''}${col.column_default ? ` DEFAULT ${col.column_default}` : ''}`);
    });

    // Check if table exists and has data
    const count = await sql`SELECT COUNT(*) as count FROM sessions`;
    console.log(`\n📊 Sessions table has ${count[0].count} rows`);

  } catch (error) {
    console.log('❌ Error checking schema:', error.message);
  }
}

checkSessionsSchema();
