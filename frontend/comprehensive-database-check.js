// Comprehensive Database and API Check
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function comprehensiveCheck() {
  console.log('🔍 COMPREHENSIVE DATABASE & API CHECK');
  console.log('=====================================\n');

  try {
    // 1. Test Database Connection
    console.log('1️⃣ TESTING DATABASE CONNECTION');
    const startTime = Date.now();
    await sql`SELECT 1 as test`;
    const connectionTime = Date.now() - startTime;
    console.log(`✅ Database connection successful (${connectionTime}ms)\n`);

    // 2. Check All Tables
    console.log('2️⃣ CHECKING ALL TABLES');
    const tables = await sql`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
      AND table_type = 'BASE TABLE'
      ORDER BY table_name
    `;

    console.log(`📋 Found ${tables.length} tables:`);
    for (const table of tables) {
      console.log(`  - ${table.table_name}`);
    }
    console.log('');

    // 3. Detailed Schema Check for Each Table
    console.log('3️⃣ DETAILED SCHEMA CHECK');
    const expectedTables = [
      'users', 'sessions', 'questions', 'stats', 'temp_questions',
      'user_reports', 'training_samples', 'community_assessments', 'cognitive_assessment_results'
    ];

    for (const tableName of expectedTables) {
      console.log(`📊 Table: ${tableName.toUpperCase()}`);

      try {
        // Get column details
        const columns = await sql`
          SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
          FROM information_schema.columns
          WHERE table_name = ${tableName}
          ORDER BY ordinal_position
        `;

        if (columns.length === 0) {
          console.log(`  ❌ Table does not exist or has no columns`);
          continue;
        }

        columns.forEach(col => {
          const nullable = col.is_nullable === 'YES' ? 'NULL' : 'NOT NULL';
          const defaultVal = col.column_default ? ` DEFAULT ${col.column_default}` : '';
          let typeInfo = col.data_type;

          if (col.character_maximum_length) typeInfo += `(${col.character_maximum_length})`;
          if (col.numeric_precision && col.data_type === 'numeric') {
            typeInfo += `(${col.numeric_precision}${col.numeric_scale ? ',' + col.numeric_scale : ''})`;
          }

          console.log(`    ${col.column_name}: ${typeInfo} ${nullable}${defaultVal}`);
        });

        // Get row count
        const countResult = await sql.unsafe(`SELECT COUNT(*) as count FROM ${tableName}`);
        console.log(`    📊 Row count: ${countResult[0].count}`);

        // Sample data (first row)
        const sampleData = await sql.unsafe(`SELECT * FROM ${tableName} LIMIT 1`);
        if (sampleData.length > 0) {
          console.log(`    📝 Sample data keys: ${Object.keys(sampleData[0]).join(', ')}`);
        } else {
          console.log(`    📝 No data in table`);
        }

      } catch (tableError) {
        console.log(`  ❌ Error checking table ${tableName}: ${tableError.message}`);
      }

      console.log('');
    }

    // 4. Check Database Constraints and Indexes
    console.log('4️⃣ CHECKING CONSTRAINTS & INDEXES');

    // Primary keys
    const primaryKeys = await sql`
      SELECT
        tc.table_name,
        kcu.column_name
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
      WHERE tc.constraint_type = 'PRIMARY KEY'
      AND tc.table_schema = 'public'
      ORDER BY tc.table_name, kcu.column_name
    `;

    console.log('🔑 Primary Keys:');
    primaryKeys.forEach(pk => {
      console.log(`  ${pk.table_name}.${pk.column_name}`);
    });

    // Foreign keys
    const foreignKeys = await sql`
      SELECT
        tc.table_name,
        kcu.column_name,
        ccu.table_name AS referenced_table,
        ccu.column_name AS referenced_column
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
      JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
        AND ccu.table_schema = tc.table_schema
      WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = 'public'
      ORDER BY tc.table_name, kcu.column_name
    `;

    console.log('\n🔗 Foreign Keys:');
    if (foreignKeys.length === 0) {
      console.log('  No foreign keys found');
    } else {
      foreignKeys.forEach(fk => {
        console.log(`  ${fk.table_name}.${fk.column_name} → ${fk.referenced_table}.${fk.referenced_column}`);
      });
    }

    // Indexes
    const indexes = await sql`
      SELECT
        schemaname,
        tablename,
        indexname,
        indexdef
      FROM pg_indexes
      WHERE schemaname = 'public'
      ORDER BY tablename, indexname
    `;

    console.log('\n📇 Indexes:');
    indexes.forEach(idx => {
      console.log(`  ${idx.tablename}: ${idx.indexname}`);
    });

    console.log('');

    // 5. Check Enums
    console.log('5️⃣ CHECKING ENUMS');
    const enums = await sql`
      SELECT
        n.nspname AS schema_name,
        t.typname AS type_name,
        string_agg(e.enumlabel, ', ' ORDER BY e.enumsortorder) AS enum_values
      FROM pg_type t
      JOIN pg_enum e ON t.oid = e.enumtypid
      JOIN pg_namespace n ON t.typnamespace = n.oid
      WHERE n.nspname = 'public'
      GROUP BY n.nspname, t.typname
      ORDER BY t.typname
    `;

    console.log('🏷️  Enum Types:');
    enums.forEach(enumType => {
      console.log(`  ${enumType.type_name}: ${enumType.enum_values}`);
    });

    console.log('');

  } catch (error) {
    console.log('❌ COMPREHENSIVE CHECK FAILED:');
    console.log(error.message);
  }
}

// API Endpoints Test
async function testApiEndpoints() {
  console.log('6️⃣ TESTING API ENDPOINTS');
  const endpoints = [
    { name: 'Profile API', url: 'http://localhost:3000/api/profile?email=test@example.com' },
    { name: 'Database User API', url: 'http://localhost:3000/api/database/user?userId=1&email=test@example.com' },
    { name: 'Community API', url: 'http://localhost:3000/api/community' },
    { name: 'Training Samples API', url: 'http://localhost:3000/api/training-samples' },
  ];

  for (const endpoint of endpoints) {
    try {
      console.log(`Testing ${endpoint.name}: ${endpoint.url}`);
      // Note: We can't actually call endpoints from Node.js in this context
      // This would require a proper HTTP client or running the Next.js dev server
      console.log(`  ⚠️  Would test this endpoint (requires running Next.js server)`);
    } catch (error) {
      console.log(`  ❌ ${endpoint.name} failed: ${error.message}`);
    }
  }
  console.log('');
}

// Run comprehensive check
comprehensiveCheck().then(() => {
  console.log('🎯 COMPREHENSIVE CHECK COMPLETED');
  return testApiEndpoints();
}).catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
