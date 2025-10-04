import { neon } from '@neondatabase/serverless';
import { trainingSamples } from '../db/schema.ts';
import { drizzle } from 'drizzle-orm/neon-http';
import { eq } from 'drizzle-orm';

const databaseUrl = process.env.DATABASE_URL || process.env.NEON_DATABASE_URL;

if (!databaseUrl) {
  console.error('❌ DATABASE_URL not found in environment variables');
  process.exit(1);
}

const sql = neon(databaseUrl);
const db = drizzle(sql);

const trainingTestData = [
  {
    sessionId: 'training_session_001',
    userId: 'demo_user',
    userInfo: {
      name: 'Training User 1',
      email: 'training1@example.com',
      age: '25',
      gender: 'male'
    },
    startedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
    completedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000 + 2 * 60 * 60 * 1000), // +2h
    totalQuestions: 12,
    answeredQuestions: 12,
    completionRate: 100,
    finalMmseScore: 28,
    overallGptScore: 8.5,
    questionResults: [
      { questionId: 1, answer: 'correct', timeSpent: 5.2 },
      { questionId: 2, answer: 'correct', timeSpent: 3.8 }
    ],
    status: 'completed',
    usageMode: 'personal',
    assessmentType: 'cognitive'
  },
  {
    sessionId: 'training_session_002',
    userId: 'community-assessments-001',
    userInfo: {
      name: 'Community User 1',
      email: 'community1@example.com',
      age: '35',
      gender: 'female'
    },
    startedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    completedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000 + 1.5 * 60 * 60 * 1000),
    totalQuestions: 12,
    answeredQuestions: 10,
    completionRate: 83,
    finalMmseScore: 24,
    overallGptScore: 7.2,
    questionResults: [
      { questionId: 1, answer: 'correct', timeSpent: 4.1 },
      { questionId: 2, answer: 'incorrect', timeSpent: 6.3 }
    ],
    status: 'completed',
    usageMode: 'community',
    assessmentType: 'cognitive'
  },
  {
    sessionId: 'training_session_003',
    userId: 'community-assessments-002',
    userInfo: {
      name: 'Community User 2',
      email: 'community2@example.com',
      age: '42',
      gender: 'male'
    },
    startedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    completedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 1.8 * 60 * 60 * 1000),
    totalQuestions: 12,
    answeredQuestions: 12,
    completionRate: 100,
    finalMmseScore: 26,
    overallGptScore: 8.0,
    questionResults: [
      { questionId: 1, answer: 'correct', timeSpent: 3.9 },
      { questionId: 2, answer: 'correct', timeSpent: 4.2 }
    ],
    status: 'completed',
    usageMode: 'community',
    assessmentType: 'cognitive'
  },
  {
    sessionId: 'training_session_004',
    userId: 'demo_user',
    userInfo: {
      name: 'Training User 1',
      email: 'training1@example.com',
      age: '25',
      gender: 'male'
    },
    startedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000), // 1 day ago
    completedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000 + 1.2 * 60 * 60 * 1000), // +1.2h
    totalQuestions: 12,
    answeredQuestions: 11,
    completionRate: 92,
    finalMmseScore: 27,
    overallGptScore: 7.8,
    questionResults: [
      { questionId: 1, answer: 'correct', timeSpent: 4.5 },
      { questionId: 2, answer: 'correct', timeSpent: 4.1 }
    ],
    status: 'completed',
    usageMode: 'personal',
    assessmentType: 'cognitive'
  },
  {
    sessionId: 'training_session_005',
    userId: 'community-assessments-003',
    userInfo: {
      name: 'Community User 3',
      email: 'community3@example.com',
      age: '38',
      gender: 'female'
    },
    startedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
    completedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 1.6 * 60 * 60 * 1000), // +1.6h
    totalQuestions: 12,
    answeredQuestions: 12,
    completionRate: 100,
    finalMmseScore: 29,
    overallGptScore: 8.7,
    questionResults: [
      { questionId: 1, answer: 'correct', timeSpent: 3.7 },
      { questionId: 2, answer: 'correct', timeSpent: 3.9 }
    ],
    status: 'completed',
    usageMode: 'community',
    assessmentType: 'cognitive'
  }
];

async function seedTrainingData() {
  console.log('🌱 Starting to seed training samples data...');

  try {
    // Check if data already exists
    const existingData = await db.select().from(trainingSamples).limit(1);
    if (existingData.length > 0) {
      console.log('⚠️ Training samples data already exists. Skipping seed operation.');
      return;
    }

    // Insert training data
    for (const data of trainingTestData) {
      console.log(`📝 Inserting training sample for session: ${data.sessionId}`);
      await db.insert(trainingSamples).values(data);
    }

    console.log(`✅ Successfully seeded ${trainingTestData.length} training samples`);

    // Verify the data was inserted
    const count = await db.$count(trainingSamples);
    console.log(`📊 Total training samples in database: ${count}`);

  } catch (error) {
    console.error('❌ Error seeding training samples:', error);
    throw error;
  }
}

// Run the seed function
seedTrainingData()
  .then(() => {
    console.log('🎉 Training samples seeding completed successfully!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Training samples seeding failed:', error);
    process.exit(1);
  });
