-- Create training_samples table with correct snake_case columns
-- Run this in your PostgreSQL database console

-- Drop table if exists (CAUTION: will lose data)
DROP TABLE IF EXISTS training_samples;

-- Create table with snake_case column names
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
);

-- Insert test data
INSERT INTO training_samples (session_id, user_id, user_email, user_name, question_id, question_text, audio_filename, auto_transcript) VALUES
('training_001', 'user_30oCOjaXrJLn9Uy1nDMaNrST1vn', 'ledinhphuc1408@gmail.com', 'Phuc Le', 1, 'What is your name?', 'audio1.wav', 'My name is Phuc'),
('training_002', 'community-assessments-001', 'ledinhphuc1408@gmail.com', 'Phuc Le Community', 2, 'What year is it?', 'audio2.wav', 'It is 2024'),
('training_003', 'user_30oCOjaXrJLn9Uy1nDMaNrST1vn', 'ledinhphuc1408@gmail.com', 'Phuc Le', 3, 'Count from 1 to 5', 'audio3.wav', 'One two three four five');

-- Verify table creation
SELECT 'Table created successfully!' as status;
SELECT COUNT(*) as total_rows FROM training_samples;
