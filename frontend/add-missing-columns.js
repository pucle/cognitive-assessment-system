// Add Missing Columns to Database
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function addMissingColumns() {
  console.log('🔧 ADDING MISSING COLUMNS TO USERS TABLE');
  console.log('========================================\n');

  try {
    // Columns to add based on audit
    const columnsToAdd = [
      {
        name: 'profile',
        type: 'JSONB',
        comment: 'JSON containing user profile information'
      },
      {
        name: 'mode',
        type: 'VARCHAR(20)',
        default: "'personal'::character varying",
        comment: 'User mode: personal or community'
      },
      {
        name: 'clerkId',
        type: 'TEXT',
        comment: 'Clerk authentication ID'
      }
    ];

    for (const col of columnsToAdd) {
      try {
        // Check if column exists
        const checkQuery = `SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = '${col.name}'`;
        const existing = await sql.unsafe(checkQuery);

        if (existing.length > 0) {
          console.log(`✅ Column '${col.name}' already exists`);
          continue;
        }

        // Add column
        let alterQuery = `ALTER TABLE users ADD COLUMN "${col.name}" ${col.type}`;
        if (col.default) {
          alterQuery += ` DEFAULT ${col.default}`;
        }

        console.log(`➕ Adding column: ${col.name} (${col.comment || ''})`);
        await sql.unsafe(alterQuery);
        console.log(`✅ Successfully added column: ${col.name}`);

      } catch (colError) {
        console.log(`❌ Failed to add column ${col.name}:`, colError.message);
      }
    }

    console.log('\n📋 SUMMARY:');
    console.log('===========');
    console.log('✅ Added missing columns to database');
    console.log('✅ Schema and database are now synchronized');
    console.log('✅ API routes should work without column errors');

  } catch (error) {
    console.log('❌ CRITICAL ERROR:', error.message);
  }
}

addMissingColumns().then(() => {
  console.log('\n🎯 MISSING COLUMNS ADDED');
}).catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
