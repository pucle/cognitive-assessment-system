// Check Foreign Key Readiness
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function analyzeRelationships() {
  console.log('🔗 ANALYZING TABLE RELATIONSHIPS');
  console.log('=================================\n');

  try {
    // Check sessions.user_id data types and values
    console.log('📊 Sessions.user_id analysis:');
    const sessionUserIds = await sql`SELECT DISTINCT user_id FROM sessions LIMIT 10`;
    console.log('Sample user_id values:', sessionUserIds.map(row => `"${row.user_id}"`).join(', '));

    // Check if user_id values are numeric or text
    const numericUserIds = await sql`SELECT COUNT(*) as count FROM sessions WHERE user_id ~ '^[0-9]+$'`;
    const textUserIds = await sql`SELECT COUNT(*) as count FROM sessions WHERE user_id IS NOT NULL AND user_id !~ '^[0-9]+$'`;
    const nullUserIds = await sql`SELECT COUNT(*) as count FROM sessions WHERE user_id IS NULL`;

    console.log(`Numeric user_ids: ${numericUserIds[0].count}`);
    console.log(`Text user_ids: ${textUserIds[0].count}`);
    console.log(`NULL user_ids: ${nullUserIds[0].count}`);

    // Check stats.user_id
    console.log('\n📊 Stats.user_id analysis:');
    const statsUserIds = await sql`SELECT DISTINCT user_id FROM stats WHERE user_id IS NOT NULL LIMIT 10`;
    console.log('Sample user_id values:', statsUserIds.map(row => `"${row.user_id}"`).join(', '));

    // Check session_id formats
    console.log('\n📊 Session IDs analysis:');
    const sessionIds = await sql`SELECT DISTINCT session_id FROM questions LIMIT 5`;
    console.log('Sample session_id values:', sessionIds.map(row => `"${row.session_id}"`).join(', '));

    const sessionIdTypes = await sql`SELECT COUNT(*) as count FROM questions WHERE session_id ~ '^[0-9]+$'`;
    console.log(`Numeric session_ids: ${sessionIdTypes[0].count}`);

    // Check if we can establish relationships
    console.log('\n🔍 RELATIONSHIP FEASIBILITY:');

    if (numericUserIds[0].count > 0) {
      // Check if numeric user_ids exist in users table
      const validUserRefs = await sql`SELECT COUNT(*) as count FROM sessions WHERE user_id ~ '^[0-9]+$' AND user_id::integer IN (SELECT id FROM users)`;
      console.log(`✅ Valid user_id references: ${validUserRefs[0].count}/${numericUserIds[0].count}`);
    }

    // Check session references
    const totalQuestions = await sql`SELECT COUNT(*) as count FROM questions`;
    const validSessionRefs = await sql`SELECT COUNT(*) as count FROM questions WHERE session_id IN (SELECT id::text FROM sessions)`;
    console.log(`✅ Valid session_id references: ${validSessionRefs[0].count}/${totalQuestions[0].count}`);

    // Recommendations
    console.log('\n💡 RECOMMENDATIONS:');
    console.log('==================');

    if (numericUserIds[0].count === 0) {
      console.log('❌ Cannot add user_id foreign keys - all values are non-numeric');
    } else if (validUserRefs[0].count < numericUserIds[0].count) {
      console.log('⚠️  Some user_id references are invalid - need data cleanup first');
    } else {
      console.log('✅ Can add user_id foreign keys');
    }

    if (validSessionRefs[0].count < totalQuestions[0].count) {
      console.log('⚠️  Some session_id references are invalid - need data cleanup first');
    } else {
      console.log('✅ Can add session_id foreign keys');
    }

  } catch (error) {
    console.log('❌ Error analyzing relationships:', error.message);
  }
}

analyzeRelationships().then(() => {
  console.log('\n🎯 RELATIONSHIP ANALYSIS COMPLETED');
}).catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
