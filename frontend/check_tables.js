const Database = require('better-sqlite3');
const db = new Database('./cognitive_assessment.db');

console.log('=== TABLES IN DATABASE ===');
try {
  const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all();
  console.log('Tables:', tables.map(t => t.name));

  // Check cognitive_assessment_results
  if (tables.find(t => t.name === 'cognitive_assessment_results')) {
    console.log('\n=== COGNITIVE_ASSESSMENT_RESULTS ===');
    const results = db.prepare('SELECT sessionId, userId, usageMode, userInfo FROM cognitive_assessment_results LIMIT 3').all();
    console.log(JSON.stringify(results, null, 2));
  }
} catch (e) {
  console.log('Error:', e.message);
}

db.close();
