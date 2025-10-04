// Test Single API Endpoint with Detailed Debugging
const http = require('http');
const https = require('https');

function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https:') ? https : http;
    const startTime = Date.now();

    console.log(`\n🔍 TESTING: ${options.method || 'GET'} ${url}`);

    const req = protocol.request(url, {
      method: options.method || 'GET',
      headers: {
        'User-Agent': 'Test-Script/1.0',
        'Accept': 'application/json',
        ...options.headers
      },
      timeout: 8000 // 8 second timeout
    }, (res) => {
      const responseTime = Date.now() - startTime;
      console.log(`📊 Response: ${res.statusCode} ${res.statusMessage} (${responseTime}ms)`);

      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        const totalTime = Date.now() - startTime;

        try {
          const json = JSON.parse(data);
          console.log(`✅ SUCCESS - JSON Response`);
          console.log(`   Response size: ${data.length} chars`);
          if (json.success !== undefined) {
            console.log(`   Success: ${json.success}`);
          }
          resolve({ status: res.statusCode, data: json, responseTime: totalTime });
        } catch (e) {
          console.log(`📄 Text Response (${data.length} chars):`);
          console.log(data.substring(0, 200) + (data.length > 200 ? '...' : ''));
          resolve({ status: res.statusCode, data, responseTime: totalTime });
        }
      });
    });

    req.on('error', (error) => {
      const errorTime = Date.now() - startTime;
      console.log(`❌ Error after ${errorTime}ms: ${error.message}`);
      reject(error);
    });

    req.on('timeout', () => {
      console.log(`⏰ Timeout after 8s`);
      req.destroy();
      reject(new Error('Request timeout'));
    });

    if (options.body) {
      req.write(JSON.stringify(options.body));
    }

    req.end();
  });
}

async function testEndpoint(name, url, options = {}) {
  try {
    const result = await makeRequest(url, options);
    return { name, success: true, status: result.status, responseTime: result.responseTime };
  } catch (error) {
    return { name, success: false, error: error.message };
  }
}

async function main() {
  console.log('🧪 SINGLE API ENDPOINT TESTS');
  console.log('=============================\n');

  // Check server first
  try {
    await makeRequest('http://localhost:3000');
    console.log('✅ Server is running\n');
  } catch (error) {
    console.log('❌ Server is not running\n');
    return;
  }

  const tests = [
    { name: 'Database User API', url: 'http://localhost:3000/api/database/user?userId=2' },
    { name: 'Profile API', url: 'http://localhost:3000/api/profile?email=test@example.com' },
    { name: 'Community API', url: 'http://localhost:3000/api/community' },
    { name: 'Training Samples API', url: 'http://localhost:3000/api/training-samples' },
  ];

  const results = [];

  for (const test of tests) {
    const result = await testEndpoint(test.name, test.url);
    results.push(result);

    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  console.log('\n📊 FINAL RESULTS:');
  console.log('================');

  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log(`✅ Successful: ${successful.length}/${results.length}`);
  successful.forEach(r => {
    console.log(`   • ${r.name}: ${r.status} (${r.responseTime}ms)`);
  });

  if (failed.length > 0) {
    console.log(`❌ Failed: ${failed.length}/${results.length}`);
    failed.forEach(r => {
      console.log(`   • ${r.name}: ${r.error}`);
    });
  }

  console.log('\n🎯 TESTING COMPLETE');
}

main().catch(console.error);
