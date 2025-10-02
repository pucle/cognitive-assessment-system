// Debug script to test greeting logic
console.log('🧪 Testing greeting logic...');

// Test cases
const testCases = [
  { name: 'Đình Phúc', age: '25', gender: 'Nam', language: 'vi' },
  { name: 'Nguyễn Văn A', age: '35', gender: 'Nam', language: 'vi' },
  { name: 'Trần Thị B', age: '65', gender: 'Nữ', language: 'vi' },
  { name: '', age: '25', gender: 'Nam', language: 'vi' },
  { name: null, age: '25', gender: 'Nam', language: 'vi' },
  { name: undefined, age: '25', gender: 'Nam', language: 'vi' }
];

function generateGreeting(data, language) {
  console.log('🔍 Testing with:', { data, language });

  // Validate data and provide fallback values
  if (!data || !data.name) {
    console.warn('❌ Invalid user data received:', data);
    return 'Chào mừng';
  }

  const nameParts = data.name.trim().split(/\s+/);
  let displayName = '';

  console.log('🔍 Name processing:', { name: data.name, nameParts, length: nameParts.length });

  // Xử lý tên theo quy tắc mới - Sửa để hiển thị đúng "Đình Phúc"
  if (nameParts.length > 2) {
    // Nếu tên có > 2 từ: Lấy 2 từ cuối của tên đầy đủ
    displayName = nameParts.slice(-2).join(' '); // Lấy 2 từ cuối
    console.log('🔍 Multiple words, displayName:', displayName);
  } else if (nameParts.length === 2) {
    // Nếu tên có 2 từ: Lấy cả 2 từ (ví dụ: "Đình Phúc")
    displayName = nameParts.join(' ');
    console.log('🔍 Two words, displayName:', displayName);
  } else if (nameParts.length === 1) {
    // Nếu chỉ có 1 từ: Lấy từ đó
    displayName = nameParts[0];
    console.log('🔍 One word, displayName:', displayName);
  } else {
    displayName = data.name; // Fallback
    console.log('🔍 Fallback, displayName:', displayName);
  }

  const age = parseInt(data.age || '25');
  let honorific = '';

  console.log('🔍 Age and gender:', { age, gender: data.gender });

  // Special cases for specific names
  const specialNames = ['Phan Nguyễn Trà Ly', 'Nguyễn Phúc Nguyên', 'Nguyễn Tâm'];
  if (specialNames.includes(data.name)) {
    honorific = 'con lợn';
    const finalGreeting = `${honorific} ${displayName}`;
    console.log('🔍 Special name case, final greeting:', finalGreeting);
    return finalGreeting;
  }

  if (age >= 60) {
    honorific = (data.gender || 'Nam') === 'Nam' ?
      (language === 'vi' ? 'ông' : 'Sir') :
      (language === 'vi' ? 'bà' : 'Madam');
  } else if (age >= 30) {
    honorific = (data.gender || 'Nam') === 'Nam' ?
      (language === 'vi' ? 'anh' : 'Mr.') :
      (language === 'vi' ? 'chị' : 'Ms.');
  } else {
    honorific = language === 'vi' ? '' : '';
  }

  const finalGreeting = `${honorific} ${displayName}`.trim();
  console.log('🔍 Normal case, final greeting:', finalGreeting);
  return finalGreeting;
}

// Run tests
testCases.forEach((testCase, index) => {
  console.log(`\n📋 Test case ${index + 1}:`);
  const result = generateGreeting(testCase, 'vi');
  console.log(`✅ Result: "${result}"`);
  console.log('─'.repeat(50));
});

console.log('\n🎯 If greeting shows "dùng mới" instead of expected, check:');
console.log('1. Is generateGreeting being called?');
console.log('2. What is the actual userData.name value?');
console.log('3. Is there another component overriding the greeting?');
