const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

// Load environment variables từ .env.local
require('dotenv').config({ path: path.join(__dirname, '.env.local') });

// Đọc connection string từ environment variables
const connectionString = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

if (!connectionString) {
  console.error('❌ DATABASE_URL not found in environment variables');
  process.exit(1);
}

const pool = new Pool({
  connectionString: connectionString,
});

async function setupDatabase() {
  try {
    console.log('🔗 Connecting to database...');
    
    // Đọc SQL file
    const sql = fs.readFileSync('./create_users_table.sql', 'utf8');
    
    // Thực thi SQL
    await pool.query(sql);
    
    console.log('✅ Users table created successfully!');
    
    // Kiểm tra bảng đã tồn tại chưa
    const result = await pool.query(`
      SELECT column_name, data_type 
      FROM information_schema.columns 
      WHERE table_name = 'users'
      ORDER BY ordinal_position;
    `);
    
    console.log('📋 Users table structure:');
    result.rows.forEach(row => {
      console.log(`  - ${row.column_name}: ${row.data_type}`);
    });
    
  } catch (error) {
    console.error('❌ Error setting up database:', error.message);
  } finally {
    await pool.end();
  }
}

setupDatabase();
