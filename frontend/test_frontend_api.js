/**
 * Frontend API Integration Test
 * Tests the connection between frontend and backend
 */

const API_BASE_URL = 'http://localhost:5001';

async function testFrontendBackendIntegration() {
    console.log('🧪 Testing Frontend-Backend Integration');
    console.log('=====================================');

    const tests = [
        {
            name: 'Backend Health Check',
            test: () => testEndpoint('/api/health', 'GET')
        },
        {
            name: 'MMSE Questions Loading',
            test: () => testEndpoint('/api/mmse/questions', 'GET')
        },
        {
            name: 'User Profile with Email',
            test: () => testEndpoint('/api/user/profile?email=test@example.com', 'GET')
        },
        {
            name: 'Assessment Queue',
            test: () => testEndpoint('/api/assess-queue', 'POST', {
                question_id: 1,
                transcript: 'Test answer for integration test',
                user_id: 'test_user',
                session_id: 'test_session_frontend',
                timestamp: new Date().toISOString()
            })
        }
    ];

    const results = [];

    for (const { name, test } of tests) {
        console.log(`\n🔍 Testing: ${name}`);
        try {
            const result = await test();
            console.log(`✅ ${name}: PASS`);
            results.push({ name, status: 'PASS', data: result });
        } catch (error) {
            console.log(`❌ ${name}: FAIL - ${error.message}`);
            results.push({ name, status: 'FAIL', error: error.message });
        }
    }

    // Summary
    console.log('\n📊 TEST SUMMARY');
    console.log('================');
    const passed = results.filter(r => r.status === 'PASS').length;
    const total = results.length;
    
    console.log(`Results: ${passed}/${total} tests passed`);
    
    results.forEach(result => {
        const status = result.status === 'PASS' ? '✅' : '❌';
        console.log(`${status} ${result.name}`);
        if (result.error) {
            console.log(`   Error: ${result.error}`);
        }
    });

    if (passed === total) {
        console.log('\n🎉 All tests passed! Frontend-backend integration is working.');
    } else {
        console.log('\n⚠️ Some tests failed. Check the backend server and API endpoints.');
    }

    return results;
}

async function testEndpoint(path, method, data = null) {
    const url = `${API_BASE_URL}${path}`;
    
    const options = {
        method,
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
    };

    if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
    }

    const response = await fetch(url, options);
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
}

// Test the fetchWithFallback function specifically
async function testFetchWithFallback() {
    console.log('\n🔧 Testing fetchWithFallback Function');
    console.log('====================================');

    // Simulate the frontend environment
    const fetchWithFallback = async (url, options = {}, fallbackData = null) => {
        try {
            console.log(`🔗 Attempting to fetch: ${url}`);
            
            const controller = new AbortController();
            const timeoutMs = url.includes('/assess') ? 60000 : 15000;
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

            const response = await fetch(url, {
                ...options,
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return response;
        } catch (error) {
            console.warn(`⚠️ Request failed: ${error.message}`);
            
            if (url.includes('/api/mmse/questions') && fallbackData) {
                console.log('📋 Using mock questions data');
                return new Response(JSON.stringify({
                    success: true,
                    data: {
                        questions: [
                            { id: 'Q1', domain: 'orientation', question_text: 'Test question 1' },
                            { id: 'Q2', domain: 'registration', question_text: 'Test question 2' }
                        ],
                        total_points: 30
                    }
                }), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' }
                });
            }

            throw error;
        }
    };

    // Test with real backend
    try {
        const response = await fetchWithFallback(`${API_BASE_URL}/api/mmse/questions`);
        const data = await response.json();
        console.log('✅ fetchWithFallback works with real backend');
        console.log(`   Loaded ${data.data?.questions?.length || 0} questions`);
    } catch (error) {
        console.log(`❌ fetchWithFallback failed: ${error.message}`);
    }

    // Test with fallback
    try {
        const response = await fetchWithFallback('http://nonexistent-server:9999/api/mmse/questions', {}, true);
        const data = await response.json();
        console.log('✅ fetchWithFallback fallback mechanism works');
        console.log(`   Fallback loaded ${data.data?.questions?.length || 0} questions`);
    } catch (error) {
        console.log(`❌ fetchWithFallback fallback failed: ${error.message}`);
    }
}

// Run all tests
async function runAllTests() {
    await testFrontendBackendIntegration();
    await testFetchWithFallback();
    
    console.log('\n🏁 All tests completed!');
    console.log('\nNext steps:');
    console.log('1. If tests passed: Your frontend should work with the backend');
    console.log('2. If tests failed: Check backend server and fix API endpoints');
    console.log('3. Update frontend .env.local with NEXT_PUBLIC_API_URL=http://localhost:5001');
}

// Export for Node.js testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { testFrontendBackendIntegration, testFetchWithFallback, runAllTests };
}

// Auto-run if called directly in browser
if (typeof window !== 'undefined') {
    runAllTests();
}
