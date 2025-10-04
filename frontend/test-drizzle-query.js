// Test Drizzle query directly
const db = require('./db/drizzle');
const { users } = require('./db/schema');
const { eq } = require('drizzle-orm');

async function testDrizzleQuery() {
  console.log('🧪 TESTING DRIZZLE QUERY DIRECTLY');

  try {
    console.log('Testing userId=2 query...');
    const result1 = await db.select().from(users).where(eq(users.id, 2));
    console.log('Result:', result1);
    console.log('Found:', result1.length, 'records');

    if (result1.length > 0) {
      console.log('User data:');
      console.log('  id:', result1[0].id, typeof result1[0].id);
      console.log('  email:', result1[0].email, typeof result1[0].email);
      console.log('  name:', result1[0].name, typeof result1[0].name);
      console.log('  displayName:', result1[0].displayName, typeof result1[0].displayName);
      console.log('  profile:', result1[0].profile, typeof result1[0].profile);
    }

    console.log('\nTesting email query...');
    const result2 = await db.select().from(users).where(eq(users.email, 'ledinhphuc1408@gmail.com'));
    console.log('Result:', result2);
    console.log('Found:', result2.length, 'records');

    if (result2.length > 0) {
      console.log('User data:');
      console.log('  id:', result2[0].id, typeof result2[0].id);
      console.log('  email:', result2[0].email, typeof result2[0].email);
      console.log('  name:', result2[0].name, typeof result2[0].name);
      console.log('  displayName:', result2[0].displayName, typeof result2[0].displayName);
      console.log('  profile:', result2[0].profile, typeof result2[0].profile);
    }

    console.log('\nTesting all users...');
    const allUsers = await db.select().from(users);
    console.log('Total users:', allUsers.length);
    allUsers.forEach((user, index) => {
      console.log(`User ${index + 1}: id=${user.id}, email=${user.email}, name=${user.name}`);
    });

  } catch (error) {
    console.log('❌ DRIZZLE QUERY ERROR:', error.message);
    console.log('Stack:', error.stack);
  }
}

testDrizzleQuery();
