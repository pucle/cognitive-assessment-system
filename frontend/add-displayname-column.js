// Add displayName column to users table
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function addDisplayNameColumn() {
  try {
    console.log('🔧 Adding displayName column to users table...');

    // Check if column already exists
    const columns = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'displayname'`;
    if (columns.length > 0) {
      console.log('✅ displayName column already exists');
      return;
    }

    // Add the column
    await sql`ALTER TABLE users ADD COLUMN "displayName" text`;
    console.log('✅ Successfully added displayName column');

    // Also add other missing columns from schema
    const missingColumns = [
      { name: 'profile', type: 'jsonb' },
      { name: 'mode', type: 'user_mode', default: "'personal'" },
      { name: 'clerkId', type: 'text' },
      { name: 'createdAt', type: 'timestamp with time zone', default: 'now()' },
      { name: 'updatedAt', type: 'timestamp with time zone', default: 'now()' }
    ];

    for (const col of missingColumns) {
      try {
        // Check if column exists using raw SQL
        const checkQuery = `SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = '${col.name}'`;
        const checkCol = await sql.unsafe(checkQuery);

        if (checkCol.length === 0) {
          const defaultClause = col.default ? ` DEFAULT ${col.default}` : '';
          const alterQuery = `ALTER TABLE users ADD COLUMN "${col.name}" ${col.type}${defaultClause}`;
          await sql.unsafe(alterQuery);
          console.log(`✅ Added ${col.name} column`);
        } else {
          console.log(`✅ ${col.name} column already exists`);
        }
      } catch (colError) {
        console.log(`⚠️  Error with column ${col.name}:`, colError.message);
      }
    }

  } catch (error) {
    console.log('❌ Error adding columns:', error.message);
  }
}

addDisplayNameColumn();
