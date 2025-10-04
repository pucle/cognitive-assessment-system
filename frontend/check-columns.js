const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });
const sql = neon(process.env.DATABASE_URL);

sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'users' ORDER BY column_name`.then(columns => {
  console.log('Current users table columns:');
  columns.forEach(col => console.log(' -', col.column_name));
}).catch(console.error);
