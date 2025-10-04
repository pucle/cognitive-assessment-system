// Add Foreign Key Constraints to Database
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function checkDataIntegrity() {
  console.log('🔍 CHECKING DATA INTEGRITY BEFORE ADDING FOREIGN KEYS');
  console.log('====================================================\n');

  try {
    // Check for orphaned records in sessions
    const orphanedSessions = await sql`SELECT COUNT(*) as count FROM sessions WHERE user_id IS NOT NULL AND user_id::integer NOT IN (SELECT id FROM users)`;
    console.log(`❌ Orphaned sessions (user_id not in users): ${orphanedSessions[0].count}`);

    // Check for orphaned records in questions
    const orphanedQuestions = await sql`SELECT COUNT(*) as count FROM questions WHERE session_id NOT IN (SELECT id::text FROM sessions)`;
    console.log(`❌ Orphaned questions (session_id not in sessions): ${orphanedQuestions[0].count}`);

    // Check for orphaned records in stats
    const orphanedStats = await sql`SELECT COUNT(*) as count FROM stats WHERE session_id NOT IN (SELECT id::text FROM sessions)`;
    console.log(`❌ Orphaned stats (session_id not in sessions): ${orphanedStats[0].count}`);

    const orphanedStatsUsers = await sql`SELECT COUNT(*) as count FROM stats WHERE user_id IS NOT NULL AND user_id::integer NOT IN (SELECT id FROM users)`;
    console.log(`❌ Orphaned stats (user_id not in users): ${orphanedStatsUsers[0].count}`);

    // Check for orphaned records in temp_questions
    const orphanedTempQuestions = await sql`SELECT COUNT(*) as count FROM temp_questions WHERE session_id NOT IN (SELECT id::text FROM sessions)`;
    console.log(`❌ Orphaned temp_questions (session_id not in sessions): ${orphanedTempQuestions[0].count}`);

    return {
      sessions: orphanedSessions[0].count,
      questions: orphanedQuestions[0].count,
      stats: orphanedStats[0].count,
      statsUsers: orphanedStatsUsers[0].count,
      tempQuestions: orphanedTempQuestions[0].count
    };

  } catch (error) {
    console.log('❌ Error checking data integrity:', error.message);
    return null;
  }
}

async function addForeignKeys() {
  console.log('\n🔧 ADDING FOREIGN KEY CONSTRAINTS');
  console.log('==================================\n');

  const constraints = [
    {
      name: 'fk_sessions_user_id',
      table: 'sessions',
      column: 'user_id',
      refTable: 'users',
      refColumn: 'id',
      onDelete: 'SET NULL'
    },
    {
      name: 'fk_questions_session_id',
      table: 'questions',
      column: 'session_id',
      refTable: 'sessions',
      refColumn: 'id',
      onDelete: 'CASCADE'
    },
    {
      name: 'fk_stats_session_id',
      table: 'stats',
      column: 'session_id',
      refTable: 'sessions',
      refColumn: 'id',
      onDelete: 'CASCADE'
    },
    {
      name: 'fk_stats_user_id',
      table: 'stats',
      column: 'user_id',
      refTable: 'users',
      refColumn: 'id',
      onDelete: 'SET NULL'
    },
    {
      name: 'fk_temp_questions_session_id',
      table: 'temp_questions',
      column: 'session_id',
      refTable: 'sessions',
      refColumn: 'id',
      onDelete: 'CASCADE'
    }
  ];

  for (const constraint of constraints) {
    try {
      // Check if constraint already exists
      const existing = await sql`
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE constraint_name = ${constraint.name}
      `;

      if (existing.length > 0) {
        console.log(`✅ Constraint ${constraint.name} already exists`);
        continue;
      }

      // Add constraint
      const query = `
        ALTER TABLE ${constraint.table}
        ADD CONSTRAINT ${constraint.name}
        FOREIGN KEY (${constraint.column})
        REFERENCES ${constraint.refTable}(${constraint.refColumn})
        ON DELETE ${constraint.onDelete}
      `;

      await sql.unsafe(query);
      console.log(`✅ Added constraint: ${constraint.name}`);

    } catch (error) {
      console.log(`❌ Failed to add ${constraint.name}: ${error.message}`);
    }
  }
}

async function main() {
  const integrity = await checkDataIntegrity();

  if (!integrity) {
    console.log('💥 Cannot proceed due to integrity check failure');
    return;
  }

  const totalOrphaned = Object.values(integrity).reduce((sum, count) => sum + count, 0);

  if (totalOrphaned > 0) {
    console.log(`\n⚠️  WARNING: Found ${totalOrphaned} orphaned records`);
    console.log('Foreign key constraints may fail. Consider cleaning up data first.');
    console.log('Proceeding anyway...\n');
  } else {
    console.log('\n✅ All data integrity checks passed');
  }

  await addForeignKeys();

  console.log('\n🎯 FOREIGN KEY SETUP COMPLETED');
}

main().catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
