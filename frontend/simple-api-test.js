// Simple API Endpoints Test
const http = require('http');
const https = require('https');

function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https:') ? https : http;
    const req = protocol.get(url, options, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          resolve({
            status: res.statusCode,
            headers: res.headers,
            data: jsonData
          });
        } catch (e) {
          resolve({
            status: res.statusCode,
            headers: res.headers,
            data: data
          });
        }
      });
    });

    req.on('error', (error) => {
      reject(error);
    });

    req.setTimeout(10000, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
  });
}

async function testApiEndpoints() {
  console.log('🚀 TESTING API ENDPOINTS');
  console.log('========================\n');

  const endpoints = [
    {
      name: 'Profile API',
      url: 'http://localhost:3000/api/profile?email=ledinhphuc1408@gmail.com',
      method: 'GET'
    },
    {
      name: 'Database User API',
      url: 'http://localhost:3000/api/database/user?userId=2&email=ledinhphuc1408@gmail.com',
      method: 'GET'
    },
    {
      name: 'Training Samples API',
      url: 'http://localhost:3000/api/training-samples',
      method: 'GET'
    },
    {
      name: 'Community API',
      url: 'http://localhost:3000/api/community',
      method: 'GET'
    }
  ];

  for (const endpoint of endpoints) {
    try {
      console.log(`📡 Testing ${endpoint.name}`);
      console.log(`   URL: ${endpoint.url}`);

      const startTime = Date.now();
      const response = await makeRequest(endpoint.url);
      const responseTime = Date.now() - startTime;

      console.log(`   Status: ${response.status} (${responseTime}ms)`);

      if (response.status >= 200 && response.status < 300) {
        console.log(`   ✅ SUCCESS`);
        if (response.data && typeof response.data === 'object') {
          console.log(`   Response: ${JSON.stringify(response.data, null, 2).substring(0, 200)}...`);
        } else {
          console.log(`   Response: ${response.data?.substring(0, 200)}...`);
        }
      } else if (response.status === 404) {
        console.log(`   ⚠️  NOT FOUND (endpoint may not exist or server not running)`);
      } else {
        console.log(`   ❌ ERROR: ${response.status}`);
        console.log(`   Response: ${JSON.stringify(response.data, null, 2)}`);
      }

    } catch (error) {
      console.log(`   ❌ FAILED: ${error.message}`);
      if (error.code === 'ECONNREFUSED') {
        console.log(`   💡 Server may not be running. Start with: npm run dev`);
      }
    }

    console.log('');
  }
}

// Check if server is running first
async function checkServerHealth() {
  console.log('🏥 CHECKING SERVER HEALTH');
  console.log('=========================\n');

  try {
    const response = await makeRequest('http://localhost:3000');
    console.log(`✅ Next.js server is running (Status: ${response.status})`);
    console.log('');
    return true;
  } catch (error) {
    console.log(`❌ Next.js server is NOT running`);
    console.log(`💡 Start the server with: npm run dev`);
    console.log('');
    return false;
  }
}

async function main() {
  const serverRunning = await checkServerHealth();
  if (serverRunning) {
    await testApiEndpoints();
  } else {
    console.log('⏭️  Skipping API tests since server is not running');
  }

  console.log('🎯 API TESTING COMPLETED');
}

main().catch(error => {
  console.log('💥 CRITICAL ERROR:', error.message);
});
