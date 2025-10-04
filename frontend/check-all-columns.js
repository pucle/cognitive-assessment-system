const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });
const sql = neon(process.env.DATABASE_URL);

(async () => {
  try {
    const result = await sql`SELECT * FROM users LIMIT 1`;
    if (result.length > 0) {
      console.log('All columns in users table:');
      console.log(Object.keys(result[0]).sort().join(', '));

      // Check specific columns
      console.log('\nChecking specific columns:');
      console.log('profile exists:', 'profile' in result[0]);
      console.log('mode exists:', 'mode' in result[0]);
      console.log('clerkId exists:', 'clerkId' in result[0]);
      console.log('createdAt exists:', 'createdAt' in result[0]);
      console.log('updatedAt exists:', 'updatedAt' in result[0]);
    } else {
      console.log('No data in users table');
    }
  } catch (e) {
    console.error('Error:', e.message);
  }
})();
