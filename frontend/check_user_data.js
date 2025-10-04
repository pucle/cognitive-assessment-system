const Database = require('better-sqlite3');
const db = new Database('./cognitive_assessment.db');

console.log('=== USERS TABLE ===');
try {
  const users = db.prepare('SELECT id, name, age, gender, email FROM users LIMIT 5').all();
  console.log(JSON.stringify(users, null, 2));
} catch (e) {
  console.log('Error querying users:', e.message);
}

console.log('\n=== COGNITIVE_ASSESSMENT_RESULTS TABLE ===');
try {
  const results = db.prepare('SELECT sessionId, userId, usageMode, json_extract(userInfo, \"$.name\") as userInfo_name FROM cognitive_assessment_results LIMIT 5').all();
  console.log(JSON.stringify(results, null, 2));
} catch (e) {
  console.log('Error querying results:', e.message);
}

db.close();
