const Database = require('./frontend/node_modules/better-sqlite3');
const db = new Database('./frontend/cognitive_assessment.db');

console.log('🔍 Checking database for real vs test data...');

try {
  const totalCount = db.prepare('SELECT COUNT(*) as c FROM cognitive_assessment_results').get();
  console.log('Total records:', totalCount.c);

  const testCount = db.prepare('SELECT COUNT(*) as c FROM cognitive_assessment_results WHERE sessionId LIKE "test_%"').get();
  console.log('Test data count:', testCount.c);

  const realCount = db.prepare('SELECT COUNT(*) as c FROM cognitive_assessment_results WHERE sessionId NOT LIKE "test_%"').get();
  console.log('Real data count:', realCount.c);

  if (realCount.c > 0) {
    console.log('\n📋 Real data samples:');
    const realSamples = db.prepare('SELECT sessionId, userId, finalMmseScore, createdAt FROM cognitive_assessment_results WHERE sessionId NOT LIKE "test_%" ORDER BY createdAt DESC LIMIT 3').all();
    realSamples.forEach((row, i) => {
      console.log(`  ${i+1}. Session: ${row.sessionId}, User: ${row.userId}, MMSE: ${row.finalMmseScore}`);
    });
  }

  // Check training_samples
  try {
    const trainingCount = db.prepare('SELECT COUNT(*) as c FROM training_samples').get();
    console.log('\n📚 Training samples count:', trainingCount.c);

    if (trainingCount.c > 0) {
      const trainingSamples = db.prepare('SELECT id, user_email, question_text FROM training_samples ORDER BY created_at DESC LIMIT 2').all();
      console.log('Training samples:');
      trainingSamples.forEach((row, i) => {
        console.log(`  ${i+1}. ID: ${row.id}, Email: ${row.user_email}`);
      });
    }
  } catch (e) {
    console.log('Training samples table may not exist');
  }

} catch (e) {
  console.error('Database error:', e.message);
} finally {
  db.close();
}

console.log('\n✅ Database check completed');