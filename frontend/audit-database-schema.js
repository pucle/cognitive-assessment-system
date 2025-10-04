// AUDIT DATABASE SCHEMA vs DRIZZLE SCHEMA
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function auditDatabaseSchema() {
  console.log('🔍 DATABASE SCHEMA AUDIT');
  console.log('=========================\n');

  try {
    // 1. Get actual database schema
    console.log('1️⃣ ACTUAL DATABASE SCHEMA (users table):');
    const dbSchema = await sql`
      SELECT
        column_name,
        data_type,
        is_nullable,
        column_default,
        character_maximum_length,
        numeric_precision
      FROM information_schema.columns
      WHERE table_name = 'users'
      ORDER BY ordinal_position
    `;

    console.log('Database columns:');
    const dbColumns = [];
    dbSchema.forEach(col => {
      const type = col.character_maximum_length
        ? `${col.data_type}(${col.character_maximum_length})`
        : col.numeric_precision
        ? `${col.data_type}(${col.numeric_precision})`
        : col.data_type;

      const nullable = col.is_nullable === 'YES' ? 'NULL' : 'NOT NULL';
      const defaultVal = col.column_default ? ` DEFAULT ${col.column_default}` : '';

      dbColumns.push(col.column_name);
      console.log(`   ${col.column_name}: ${type} ${nullable}${defaultVal}`);
    });

    console.log('\nDatabase column names:', dbColumns.join(', '));

    // 2. Check Drizzle schema (from file)
    console.log('\n2️⃣ DRIZZLE SCHEMA DEFINITION:');
    const fs = require('fs');
    const schemaPath = './db/schema.ts';

    if (fs.existsSync(schemaPath)) {
      const schemaContent = fs.readFileSync(schemaPath, 'utf8');

      // Extract users table definition
      const usersTableMatch = schemaContent.match(/export const users = pgTable\("users", \{([\s\S]*?)\}\);/);
      if (usersTableMatch) {
        console.log('Drizzle users table definition:');
        console.log(usersTableMatch[1].trim());

        // Extract column names from Drizzle schema
        const drizzleColumns = [];
        const columnMatches = usersTableMatch[1].match(/\s+(\w+):\s+\w+\([^)]*\)/g);
        if (columnMatches) {
          columnMatches.forEach(match => {
            const colName = match.trim().split(':')[0].trim();
            drizzleColumns.push(colName);
          });
        }

        console.log('\nDrizzle column names:', drizzleColumns.join(', '));

        // 3. Compare schemas
        console.log('\n3️⃣ SCHEMA COMPARISON:');

        const missingInDb = drizzleColumns.filter(col => !dbColumns.includes(col));
        const extraInDb = dbColumns.filter(col => !drizzleColumns.includes(col));

        if (missingInDb.length > 0) {
          console.log('❌ Columns in Drizzle but MISSING in database:');
          missingInDb.forEach(col => console.log(`   - ${col}`));
        } else {
          console.log('✅ All Drizzle columns exist in database');
        }

        if (extraInDb.length > 0) {
          console.log('⚠️  Extra columns in database (not in Drizzle):');
          extraInDb.forEach(col => console.log(`   - ${col}`));
        } else {
          console.log('✅ No extra columns in database');
        }

        // 4. Check the problematic "profile" column
        console.log('\n4️⃣ PROFILE COLUMN ANALYSIS:');
        const hasProfileInDb = dbColumns.includes('profile');
        const hasProfileInDrizzle = drizzleColumns.includes('profile');

        console.log(`   Database has 'profile' column: ${hasProfileInDb ? '✅ YES' : '❌ NO'}`);
        console.log(`   Drizzle schema has 'profile' column: ${hasProfileInDrizzle ? '✅ YES' : '❌ NO'}`);

        if (hasProfileInDrizzle && !hasProfileInDb) {
          console.log('\n🚨 CRITICAL: Drizzle queries "profile" but database doesn\'t have it!');
          console.log('   This causes the NeonDbError: column "profile" does not exist');

          // 5. Generate fix
          console.log('\n5️⃣ IMMEDIATE FIXES NEEDED:');

          console.log('\nA. Update Drizzle schema to remove profile column:');
          console.log('   // Comment out or remove this line:');
          console.log('   // profile: jsonb("profile"),');

          console.log('\nB. Update API queries to exclude profile:');
          console.log('   // Remove profile from select queries');

          console.log('\nC. Alternative: Add profile column to database:');
          console.log('   ALTER TABLE users ADD COLUMN profile JSONB;');
        }

      } else {
        console.log('❌ Could not extract users table from Drizzle schema');
      }
    } else {
      console.log('❌ Drizzle schema file not found');
    }

    // 6. Test problematic query
    console.log('\n6️⃣ TESTING PROBLEMATIC QUERY:');
    try {
      console.log('Testing full SELECT * query...');
      const fullQuery = await sql`SELECT * FROM users LIMIT 1`;
      console.log('✅ Full query works, columns available:', Object.keys(fullQuery[0] || {}));

      console.log('\nTesting problematic profile query...');
      const profileQuery = await sql`SELECT profile FROM users LIMIT 1`;
      console.log('❌ Profile query failed as expected');
    } catch (error) {
      console.log('❌ Profile query error:', error.message);
    }

  } catch (error) {
    console.log('❌ AUDIT ERROR:', error.message);
  }
}

// Also check API routes using users table
async function checkApiRoutes() {
  console.log('\n7️⃣ API ROUTES AUDIT:');
  console.log('=====================');

  const fs = require('fs');
  const path = require('path');

  function scanDirectory(dir, results = []) {
    const files = fs.readdirSync(dir);

    for (const file of files) {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        scanDirectory(fullPath, results);
      } else if (file === 'route.ts' || file === 'route.js') {
        const content = fs.readFileSync(fullPath, 'utf8');
        if (content.includes('users') && content.includes('profile')) {
          results.push({
            file: fullPath,
            hasUsers: true,
            hasProfile: true
          });
        } else if (content.includes('users')) {
          results.push({
            file: fullPath,
            hasUsers: true,
            hasProfile: false
          });
        }
      }
    }

    return results;
  }

  try {
    const apiRoutes = scanDirectory('./app/api');
    console.log(`Found ${apiRoutes.length} API route files using users table:`);

    apiRoutes.forEach(route => {
      const status = route.hasProfile ? '❌ HAS PROFILE REFERENCE' : '✅ OK';
      console.log(`   ${status}: ${route.file}`);
    });

  } catch (error) {
    console.log('❌ API routes scan error:', error.message);
  }
}

async function main() {
  await auditDatabaseSchema();
  await checkApiRoutes();

  console.log('\n🎯 AUDIT COMPLETE');
  console.log('==================');
  console.log('Next steps:');
  console.log('1. Fix Drizzle schema to remove profile column');
  console.log('2. Update API routes to not query profile');
  console.log('3. Test all fixes');
  console.log('4. Consider adding profile column to database if needed');
}

main().catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
