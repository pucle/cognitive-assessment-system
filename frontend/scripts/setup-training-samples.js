import { neon } from '@neondatabase/serverless';

const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

if (!databaseUrl) {
  console.error('❌ DATABASE_URL not found in environment variables');
  process.exit(1);
}

const sql = neon(databaseUrl);

async function setupTrainingSamples() {
  try {
    console.log('🔧 Setting up training_samples table...');

    // Drop table if exists
    console.log('🗑️ Dropping existing table if exists...');
    await sql`DROP TABLE IF EXISTS training_samples`;

    // Create table with correct structure
    console.log('📋 Creating training_samples table...');
    await sql`
      CREATE TABLE training_samples (
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
      )
    `;

    // Insert test data
    console.log('📝 Inserting test data...');
    await sql`
      INSERT INTO training_samples (session_id, user_id, user_email, user_name, question_id, question_text, audio_filename, auto_transcript) VALUES
      ('training_001', 'user_30oCOjaXrJLn9Uy1nDMaNrST1vn', 'ledinhphuc1408@gmail.com', 'Phuc Le', 1, 'What is your name?', 'audio1.wav', 'My name is Phuc'),
      ('training_002', 'community-assessments-001', 'ledinhphuc1408@gmail.com', 'Phuc Le Community', 2, 'What year is it?', 'audio2.wav', 'It is 2024'),
      ('training_003', 'user_30oCOjaXrJLn9Uy1nDMaNrST1vn', 'ledinhphuc1408@gmail.com', 'Phuc Le', 3, 'Count from 1 to 5', 'audio3.wav', 'One two three four five')
    `;

    // Verify data
    console.log('✅ Verifying data...');
    const count = await sql`SELECT COUNT(*) as total FROM training_samples`;
    const sample = await sql`SELECT * FROM training_samples LIMIT 1`;

    console.log(`📊 Total rows: ${count[0].total}`);
    console.log('📋 Sample row:', sample[0]);

    console.log('🎉 Training samples setup completed successfully!');

  } catch (error) {
    console.error('❌ Error setting up training samples:', error);
    console.error('Error details:', error.message);
    process.exit(1);
  }
}

// Run setup
setupTrainingSamples();
