import { neon } from '@neondatabase/serverless';
import { sql } from 'drizzle-orm';

const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

if (!databaseUrl) {
  console.error('❌ DATABASE_URL not found in environment variables');
  process.exit(1);
}

const db = neon(databaseUrl);

async function debugTrainingTable() {
  try {
    console.log('🔍 Checking training_samples table structure...');

    // Check if table exists
    const tableExists = await db(sql`
      SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = 'training_samples'
      );
    `);
    console.log('📋 Table exists:', tableExists[0]?.exists || false);

    if (!tableExists[0]?.exists) {
      console.log('❌ Table training_samples does not exist!');
      console.log('🔧 Creating table with correct structure...');

      // Create the table with snake_case column names
      await db(sql`
        CREATE TABLE IF NOT EXISTS training_samples (
          id SERIAL PRIMARY KEY,
          session_id VARCHAR(255) NOT NULL,
          user_id VARCHAR(255) NOT NULL,
          user_email VARCHAR(255),
          user_name VARCHAR(255),
          question_id INTEGER,
          question_text TEXT,
          audio_filename VARCHAR(255),
          audio_url TEXT,
          auto_transcript TEXT,
          manual_transcript TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
      `);

      console.log('✅ Table created successfully!');

      // Insert some test data
      console.log('📝 Inserting test data...');
      await db(sql`
        INSERT INTO training_samples (session_id, user_id, user_email, user_name, question_id, question_text, audio_filename, auto_transcript) VALUES
        ('training_001', 'user_30oCOjaXrJLn9Uy1nDMaNrST1vn', 'ledinhphuc1408@gmail.com', 'Phuc Le', 1, 'What is your name?', 'audio1.wav', 'My name is Phuc'),
        ('training_002', 'community-assessments-001', 'ledinhphuc1408@gmail.com', 'Phuc Le Community', 2, 'What year is it?', 'audio2.wav', 'It is 2024'),
        ('training_003', 'user_30oCOjaXrJLn9Uy1nDMaNrST1vn', 'ledinhphuc1408@gmail.com', 'Phuc Le', 3, 'Count from 1 to 5', 'audio3.wav', 'One two three four five');
      `);

      console.log('✅ Test data inserted!');
    }

    // Get table structure
    const columns = await db(sql`
      SELECT column_name, data_type, is_nullable, column_default
      FROM information_schema.columns
      WHERE table_name = 'training_samples'
      ORDER BY ordinal_position;
    `);

    console.log('🗂️ Training samples columns:');
    columns.forEach(col => {
      console.log(`- ${col.column_name}: ${col.data_type} (${col.is_nullable ? 'nullable' : 'not null'})`);
    });

    // Check sample data
    const sampleData = await db(sql`
      SELECT * FROM training_samples LIMIT 3;
    `);
    console.log('📊 Sample data (first 3 rows):');
    sampleData.forEach((row, index) => {
      console.log(`Row ${index + 1}:`, {
        id: row.id,
        session_id: row.session_id,
        user_id: row.user_id,
        user_email: row.user_email,
        user_name: row.user_name,
        question_id: row.question_id,
        created_at: row.created_at
      });
    });

    console.log(`📈 Total rows: ${sampleData.length}`);

  } catch (error) {
    console.error('❌ Error checking training_samples:', error);
    console.error('Error details:', error.message);
  }
}

// Run the debug function
debugTrainingTable()
  .then(() => {
    console.log('🎉 Debug completed!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Debug failed:', error);
    process.exit(1);
  });
