// Test script để kiểm tra profile API trực tiếp (bypass authentication)
const testProfileDirect = async () => {
  try {
    console.log('🧪 Testing Profile API Direct...');
    
    // Test data
    const testData = {
      name: "Đình Phúc",
      age: "25",
      gender: "Nam", 
      phone: "0123456789",
      title: "Test User",
      imageSrc: "",
      mmseScore: 0
    };
    
    console.log('📤 Sending test data:', testData);
    
    // Test với mock authentication headers
    const response = await fetch('http://localhost:3000/api/profile', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-clerk-auth-status': 'signed-in',
        'x-clerk-user-id': 'test-user-123',
        'x-clerk-email': 'test@example.com'
      },
      body: JSON.stringify(testData)
    });
    
    console.log('📥 Response status:', response.status);
    console.log('📥 Response headers:', Object.fromEntries(response.headers.entries()));
    
    if (response.status === 404) {
      console.log('❌ API endpoint not found - có thể do routing issue');
      return;
    }
    
    const result = await response.json();
    console.log('📥 Response body:', result);
    
    if (response.ok) {
      console.log('✅ Profile API test successful!');
    } else {
      console.log('❌ Profile API test failed!');
      console.log('Error:', result.error);
    }
    
  } catch (error) {
    console.error('❌ Test error:', error.message);
  }
};

// Test GET endpoint
const testProfileGet = async () => {
  try {
    console.log('🧪 Testing Profile GET API...');
    
    const response = await fetch('http://localhost:3000/api/profile', {
      method: 'GET',
      headers: {
        'x-clerk-auth-status': 'signed-in',
        'x-clerk-user-id': 'test-user-123',
        'x-clerk-email': 'test@example.com'
      }
    });
    
    console.log('📥 GET Response status:', response.status);
    
    if (response.status === 404) {
      console.log('❌ GET API endpoint not found');
      return;
    }
    
    const result = await response.json();
    console.log('📥 GET Response body:', result);
    
  } catch (error) {
    console.error('❌ GET Test error:', error.message);
  }
};

// Chạy tests
console.log('🚀 Starting Profile API Tests...\n');
testProfileGet().then(() => {
  console.log('\n' + '='.repeat(50) + '\n');
  testProfileDirect();
});
