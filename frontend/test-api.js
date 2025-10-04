// Simple test for API endpoints
const BASE_URL = 'http://localhost:3000';

async function testAPI() {
  console.log('🧪 Testing API endpoints...\n');

  try {
    // Test 1: Profile API (should return 401 without auth)
    console.log('1. Testing /api/profile (unauthorized)...');
    const profileResponse = await fetch(`${BASE_URL}/api/profile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({})
    });

    console.log(`   Status: ${profileResponse.status}`);
    if (profileResponse.status === 401) {
      console.log('   ✅ Correctly returns 401 Unauthorized');
    } else {
      console.log('   ❌ Expected 401, got:', profileResponse.status);
      const text = await profileResponse.text();
      console.log('   Response:', text);
    }

  } catch (error) {
    console.log('   ❌ Network error:', error.message);
  }

  try {
    // Test 2: GET Profile API
    console.log('\n2. Testing GET /api/profile...');
    const getProfileResponse = await fetch(`${BASE_URL}/api/profile`);

    console.log(`   Status: ${getProfileResponse.status}`);
    if (getProfileResponse.status === 401) {
      console.log('   ✅ Correctly returns 401 Unauthorized');
    } else {
      console.log('   ❌ Expected 401, got:', getProfileResponse.status);
    }

  } catch (error) {
    console.log('   ❌ Network error:', error.message);
  }

  console.log('\n🏁 Test completed!');
  console.log('\n📝 Note: These tests verify the API routes are working without authentication.');
  console.log('   To test with authentication, you need to sign in to Clerk first.');
}

// Run the test
testAPI().catch(console.error);
