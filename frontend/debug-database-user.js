// Debug Database User API Endpoint
const { neon } = require('@neondatabase/serverless');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function debugDatabaseUserEndpoint() {
  console.log('🔍 DEBUGGING DATABASE/USER ENDPOINT');
  console.log('===================================\n');

  try {
    // 1. Check users table schema
    console.log('1️⃣ USERS TABLE SCHEMA:');
    const schema = await sql`
      SELECT column_name, data_type, is_nullable, column_default
      FROM information_schema.columns
      WHERE table_name = 'users'
      ORDER BY ordinal_position
    `;

    schema.forEach(col => {
      console.log(`   ${col.column_name}: ${col.data_type} ${col.is_nullable === 'YES' ? 'NULL' : 'NOT NULL'}`);
    });
    console.log('');

    // 2. Check sample data in users table
    console.log('2️⃣ USERS TABLE SAMPLE DATA:');
    const sampleData = await sql`SELECT * FROM users LIMIT 3`;
    console.log(`   Found ${sampleData.length} user records`);

    if (sampleData.length > 0) {
      sampleData.forEach((user, index) => {
        console.log(`   User ${index + 1}:`);
        console.log(`     id: ${user.id} (${typeof user.id})`);
        console.log(`     email: ${user.email} (${typeof user.email})`);
        console.log(`     name: ${user.name} (${typeof user.name})`);
        console.log(`     displayName: ${user.displayName} (${typeof user.displayName})`);
        console.log('');
      });
    } else {
      console.log('   ❌ No user data found in database');
      console.log('   💡 This explains why API returns "User not found"');
    }

    // 3. Test the exact query used in the API
    console.log('3️⃣ TESTING API QUERY LOGIC:');

    // Test userId query (userId=2)
    console.log('   Testing userId=2 query:');
    try {
      const userById = await sql`SELECT * FROM users WHERE id = 2`;
      console.log(`   Found ${userById.length} users with id=2`);
      if (userById.length === 0) {
        console.log('   ❌ No user found with id=2');
      }
    } catch (error) {
      console.log(`   ❌ Query error: ${error.message}`);
    }

    // Test email query
    console.log('   Testing email query:');
    try {
      const userByEmail = await sql`SELECT * FROM users WHERE email = 'ledinhphuc1408@gmail.com'`;
      console.log(`   Found ${userByEmail.length} users with email=ledinhphuc1408@gmail.com`);
      if (userByEmail.length === 0) {
        console.log('   ❌ No user found with that email');
      }
    } catch (error) {
      console.log(`   ❌ Query error: ${error.message}`);
    }

    // 4. Check if the route file logic matches schema
    console.log('4️⃣ CHECKING ROUTE LOGIC vs SCHEMA:');
    console.log('   Route expects: user.id (integer), user.email (string)');
    console.log('   Database has: id (integer), email (string)');

    // Test Drizzle query simulation
    console.log('   Testing Drizzle-style queries:');
    try {
      // Simulate: eq(users.id, parseInt(userId))
      const drizzleIdQuery = await sql`SELECT * FROM users WHERE id = ${2}`;
      console.log(`   Drizzle id query result: ${drizzleIdQuery.length} records`);

      // Simulate: eq(users.email, email)
      const drizzleEmailQuery = await sql`SELECT * FROM users WHERE email = ${'ledinhphuc1408@gmail.com'}`;
      console.log(`   Drizzle email query result: ${drizzleEmailQuery.length} records`);

    } catch (error) {
      console.log(`   ❌ Drizzle simulation error: ${error.message}`);
    }

    // 5. Recommendations
    console.log('5️⃣ RECOMMENDATIONS:');
    if (sampleData.length === 0) {
      console.log('   ❌ CRITICAL: No user data in database');
      console.log('   💡 Solution: Insert test user data or fix database seeding');
    } else {
      console.log('   ✅ User data exists');

      const hasId2 = sampleData.some(user => user.id === 2);
      const hasTestEmail = sampleData.some(user => user.email === 'ledinhphuc1408@gmail.com');

      if (!hasId2) {
        console.log('   ⚠️  No user with id=2 (API test uses userId=2)');
      }
      if (!hasTestEmail) {
        console.log('   ⚠️  No user with email=ledinhphuc1408@gmail.com');
      }

      if (hasId2 && hasTestEmail) {
        console.log('   ✅ Test data available - API should work');
      }
    }

    console.log('   🔧 If API still fails, check:');
    console.log('      1. Drizzle schema mapping in route.ts');
    console.log('      2. Database connection in route.ts');
    console.log('      3. Error handling logic');

  } catch (error) {
    console.log('❌ DEBUG ERROR:', error.message);
  }
}

// Test the actual API endpoint
async function testApiEndpoint() {
  console.log('\n6️⃣ TESTING ACTUAL API ENDPOINT:');

  const http = require('http');

  function makeRequest(url) {
    return new Promise((resolve) => {
      const req = http.get(url, (res) => {
        let data = '';
        res.on('data', (chunk) => data += chunk);
        res.on('end', () => {
          try {
            const json = JSON.parse(data);
            resolve({ status: res.statusCode, data: json });
          } catch (e) {
            resolve({ status: res.statusCode, data });
          }
        });
      });
      req.on('error', (error) => resolve({ error: error.message }));
      req.setTimeout(5000, () => {
        req.destroy();
        resolve({ error: 'Timeout' });
      });
    });
  }

  const result = await makeRequest('http://localhost:3000/api/database/user?userId=2');
  console.log(`   API Response: ${result.status} - ${JSON.stringify(result.data)}`);
}

// Run debug
debugDatabaseUserEndpoint().then(() => {
  return testApiEndpoint();
}).then(() => {
  console.log('\n🎯 DATABASE/USER DEBUG COMPLETE');
}).catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
