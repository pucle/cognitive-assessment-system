// Comprehensive API Endpoints Debug Script
const http = require('http');
const https = require('https');

function makeDetailedRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https:') ? https : http;
    const startTime = Date.now();

    console.log(`\n🔍 REQUEST: ${options.method || 'GET'} ${url}`);
    console.log(`⏰ Start Time: ${new Date().toISOString()}`);

    const req = protocol.request(url, {
      method: options.method || 'GET',
      headers: {
        'User-Agent': 'CognitiveAssessment-Debug/1.0',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...options.headers
      },
      timeout: 15000 // 15 second timeout
    }, (res) => {
      const responseTime = Date.now() - startTime;
      console.log(`📊 RESPONSE: ${res.statusCode} ${res.statusMessage} (${responseTime}ms)`);

      let data = '';
      const headers = res.headers;

      console.log(`📋 Headers:`);
      Object.entries(headers).forEach(([key, value]) => {
        console.log(`   ${key}: ${value}`);
      });

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        const endTime = Date.now();
        const totalTime = endTime - startTime;

        console.log(`⏱️  Total Time: ${totalTime}ms`);
        console.log(`📏 Content-Length: ${data.length} bytes`);

        try {
          const jsonData = JSON.parse(data);
          console.log(`📄 JSON Response (${typeof jsonData}):`);
          console.log(JSON.stringify(jsonData, null, 2));
          resolve({
            status: res.statusCode,
            headers,
            data: jsonData,
            responseTime,
            totalTime,
            isJson: true
          });
        } catch (e) {
          console.log(`📄 Text Response (${data.length} chars):`);
          console.log(data.substring(0, 500) + (data.length > 500 ? '...' : ''));
          resolve({
            status: res.statusCode,
            headers,
            data: data,
            responseTime,
            totalTime,
            isJson: false
          });
        }
      });
    });

    req.on('error', (error) => {
      const errorTime = Date.now() - startTime;
      console.log(`❌ REQUEST ERROR after ${errorTime}ms:`);
      console.log(`   Code: ${error.code}`);
      console.log(`   Message: ${error.message}`);
      reject(error);
    });

    req.on('timeout', () => {
      console.log(`⏰ REQUEST TIMEOUT after 15s`);
      req.destroy();
      reject(new Error('Request timeout'));
    });

    if (options.body) {
      req.write(JSON.stringify(options.body));
    }

    req.end();
  });
}

async function debugAllEndpoints() {
  console.log('🚀 COMPREHENSIVE API ENDPOINTS DEBUG');
  console.log('=====================================\n');

  // Check server health first
  try {
    console.log('🏥 CHECKING SERVER HEALTH');
    await makeDetailedRequest('http://localhost:3000');
    console.log('✅ Server is responding\n');
  } catch (error) {
    console.log('❌ Server is NOT responding\n');
    console.log('💡 Make sure Next.js dev server is running: npm run dev\n');
    return;
  }

  // Test all endpoints
  const endpoints = [
    {
      name: 'Profile API - Email',
      url: 'http://localhost:3000/api/profile?email=ledinhphuc1408@gmail.com'
    },
    {
      name: 'Profile API - No Params',
      url: 'http://localhost:3000/api/profile'
    },
    {
      name: 'Database User API - UserID',
      url: 'http://localhost:3000/api/database/user?userId=2'
    },
    {
      name: 'Database User API - Email',
      url: 'http://localhost:3000/api/database/user?email=ledinhphuc1408@gmail.com'
    },
    {
      name: 'Database User API - Both Params',
      url: 'http://localhost:3000/api/database/user?userId=2&email=ledinhphuc1408@gmail.com'
    },
    {
      name: 'Database User API - POST',
      url: 'http://localhost:3000/api/database/user',
      method: 'POST',
      body: { action: 'get_user', email: 'ledinhphuc1408@gmail.com' }
    },
    {
      name: 'Training Samples API',
      url: 'http://localhost:3000/api/training-samples'
    },
    {
      name: 'Community API',
      url: 'http://localhost:3000/api/community'
    },
    {
      name: 'Community API - Pagination',
      url: 'http://localhost:3000/api/community?limit=10&offset=0'
    },
    {
      name: 'Community Assessments API',
      url: 'http://localhost:3000/api/community-assessments'
    },
    {
      name: 'Non-existent API',
      url: 'http://localhost:3000/api/non-existent'
    }
  ];

  const results = {
    working: [],
    issues: [],
    total: endpoints.length
  };

  for (const endpoint of endpoints) {
    try {
      const response = await makeDetailedRequest(endpoint.url, {
        method: endpoint.method,
        headers: endpoint.body ? { 'Content-Type': 'application/json' } : {},
        body: endpoint.body
      });

      if (response.status >= 200 && response.status < 300) {
        results.working.push({
          name: endpoint.name,
          status: response.status,
          responseTime: response.responseTime
        });
        console.log(`✅ ${endpoint.name} - WORKING\n`);
      } else {
        results.issues.push({
          name: endpoint.name,
          status: response.status,
          issue: `HTTP ${response.status}`,
          responseTime: response.responseTime
        });
        console.log(`❌ ${endpoint.name} - ISSUE: HTTP ${response.status}\n`);
      }

    } catch (error) {
      results.issues.push({
        name: endpoint.name,
        status: 'ERROR',
        issue: error.message,
        responseTime: 0
      });
      console.log(`💥 ${endpoint.name} - ERROR: ${error.message}\n`);
    }

    // Small delay between requests
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  // Summary
  console.log('📊 DEBUG SUMMARY');
  console.log('================');

  console.log(`\n✅ WORKING ENDPOINTS (${results.working.length}/${results.total}):`);
  results.working.forEach(endpoint => {
    console.log(`   • ${endpoint.name} (${endpoint.status}, ${endpoint.responseTime}ms)`);
  });

  console.log(`\n❌ ISSUES FOUND (${results.issues.length}/${results.total}):`);
  results.issues.forEach(endpoint => {
    console.log(`   • ${endpoint.name} - ${endpoint.issue} (${endpoint.responseTime}ms)`);
  });

  const successRate = ((results.working.length / results.total) * 100).toFixed(1);
  console.log(`\n🎯 SUCCESS RATE: ${successRate}%`);

  // Specific analysis for database/user endpoint
  console.log('\n🔍 DETAILED ANALYSIS - DATABASE/USER ENDPOINT');
  console.log('===============================================');

  const dbUserIssues = results.issues.filter(issue =>
    issue.name.includes('Database User API')
  );

  if (dbUserIssues.length > 0) {
    console.log('❌ All database/user endpoint variants failed with 404');
    console.log('🔧 POSSIBLE CAUSES:');
    console.log('   1. Route file not found or not properly exported');
    console.log('   2. Next.js app router cache issue');
    console.log('   3. File path mismatch: /app/api/database/user/route.ts');
    console.log('   4. Runtime error preventing route registration');
    console.log('   5. Missing file or incorrect export');

    // Check if route file exists by trying to read it
    console.log('\n🔍 VERIFYING ROUTE FILE EXISTENCE:');
    try {
      const fs = require('fs');
      const routePath = './app/api/database/user/route.ts';
      if (fs.existsSync(routePath)) {
        console.log('✅ Route file exists at:', routePath);
        const stats = fs.statSync(routePath);
        console.log('   File size:', stats.size, 'bytes');
        console.log('   Modified:', stats.mtime.toISOString());
      } else {
        console.log('❌ Route file NOT found at:', routePath);
      }
    } catch (error) {
      console.log('⚠️  Could not check file existence:', error.message);
    }
  }

  console.log('\n🎯 DEBUG COMPLETE - Check results above for specific issues');
}

// Run the debug
debugAllEndpoints().catch(error => {
  console.log('💥 CRITICAL DEBUG ERROR:', error.message);
  process.exit(1);
});
