// Test script để kiểm tra API mà không cần authentication
const testWithoutAuth = async () => {
  try {
    console.log('🧪 Testing API without authentication...');
    
    // Test các endpoint khác nhau
    const endpoints = [
      'http://localhost:3000/api/profile',
      'http://localhost:3000/api/database/user',
      'http://localhost:3000/api/health'
    ];
    
    for (const endpoint of endpoints) {
      console.log(`\n📡 Testing: ${endpoint}`);
      
      try {
        const response = await fetch(endpoint, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        console.log(`   Status: ${response.status}`);
        console.log(`   Headers:`, Object.fromEntries(response.headers.entries()));
        
        if (response.status === 200) {
          const result = await response.json();
          console.log(`   Response:`, result);
        } else {
          const text = await response.text();
          console.log(`   Error Response:`, text.substring(0, 200));
        }
        
      } catch (error) {
        console.log(`   Error:`, error.message);
      }
    }
    
  } catch (error) {
    console.error('❌ Test error:', error.message);
  }
};

// Chạy test
testWithoutAuth();
