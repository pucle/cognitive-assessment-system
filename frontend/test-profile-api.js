// Test script để kiểm tra test-profile API
const testProfileAPI = async () => {
  try {
    console.log('🧪 Testing Test Profile API...');
    
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
    
    // Test POST request
    const response = await fetch('http://localhost:3000/api/test-profile', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });
    
    console.log('📥 Response status:', response.status);
    console.log('📥 Response headers:', Object.fromEntries(response.headers.entries()));
    
    const result = await response.json();
    console.log('📥 Response body:', result);
    
    if (response.ok) {
      console.log('✅ Test Profile API test successful!');
      
      // Test GET request
      console.log('\n🧪 Testing GET request...');
      const getResponse = await fetch('http://localhost:3000/api/test-profile', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      const getResult = await getResponse.json();
      console.log('📥 GET Response:', getResult);
      
    } else {
      console.log('❌ Test Profile API test failed!');
      console.log('Error:', result.error);
    }
    
  } catch (error) {
    console.error('❌ Test error:', error.message);
  }
};

// Chạy test
testProfileAPI();