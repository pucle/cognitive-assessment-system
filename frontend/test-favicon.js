// Simple test for favicon and server status
const BASE_URL = 'http://localhost:3000';

async function testServer() {
  console.log('🧪 Testing Next.js server...\n');

  try {
    // Test 1: Homepage
    console.log('1. Testing homepage...');
    const homeResponse = await fetch(`${BASE_URL}/`);
    console.log(`   Status: ${homeResponse.status}`);
    if (homeResponse.status === 200) {
      console.log('   ✅ Homepage working');
    } else {
      console.log('   ❌ Homepage not working');
    }

    // Test 2: Favicon
    console.log('\n2. Testing favicon...');
    const faviconResponse = await fetch(`${BASE_URL}/favicon.ico`);
    console.log(`   Status: ${faviconResponse.status}`);
    if (faviconResponse.status === 200) {
      console.log('   ✅ Favicon accessible');
    } else {
      console.log('   ❌ Favicon not accessible');
      console.log(`   Response: ${faviconResponse.status} ${faviconResponse.statusText}`);
    }

    // Test 3: API route
    console.log('\n3. Testing API route...');
    const apiResponse = await fetch(`${BASE_URL}/api/profile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{"name":"Test User","age":"25","gender":"Male"}'
    });
    console.log(`   Status: ${apiResponse.status}`);
    if (apiResponse.status === 200) {
      console.log('   ✅ API working correctly');
      const data = await apiResponse.json();
      console.log('   Response:', JSON.stringify(data, null, 2));
    } else if (apiResponse.status === 404) {
      console.log('   ❌ API route not found');
    } else {
      console.log(`   ⚠️ API returned ${apiResponse.status}`);
      const errorText = await apiResponse.text();
      console.log('   Error:', errorText);
    }

  } catch (error) {
    console.log('   ❌ Network error:', error.message);
    console.log('\n💡 Make sure the Next.js dev server is running:');
    console.log('   npm run dev');
  }

  console.log('\n🏁 Test completed!');
}

// Run the test
testServer().catch(console.error);
