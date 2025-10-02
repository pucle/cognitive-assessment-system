// Test script for force next question functionality
console.log('🧪 Testing Force Next Question Functionality');
console.log('=' * 50);

// Mock state for testing
let currentQuestionIndex = 0;
const questions = [
  { id: '1', text: 'Question 1' },
  { id: '2', text: 'Question 2' },
  { id: '3', text: 'Question 3' }
];

function testAutoProgression() {
  console.log('\n📋 Test Case: Auto-progression in normal mode');
  console.log(`Current question index: ${currentQuestionIndex}`);
  console.log(`Total questions: ${questions.length}`);

  // Simulate submitDomainAssessment completion
  setTimeout(() => {
    const nextQuestionIndex = currentQuestionIndex + 1;
    console.log(`\n⏰ After 1.5s delay:`);
    console.log(`Next question index: ${nextQuestionIndex}`);

    if (nextQuestionIndex < questions.length) {
      console.log(`✅ Auto-progressing to question ${nextQuestionIndex + 1}/${questions.length}`);
      console.log(`📝 Question: ${questions[nextQuestionIndex].text}`);

      // Update state (simulated)
      currentQuestionIndex = nextQuestionIndex;
      console.log(`🔄 Updated currentQuestionIndex to: ${currentQuestionIndex}`);
      console.log(`🎯 Now on question ${currentQuestionIndex + 1}/${questions.length}`);
    } else {
      console.log('🎉 All questions completed!');
    }
  }, 1500);

  console.log('⏳ Waiting for auto-progression...');
}

function testTrainingMode() {
  console.log('\n📋 Test Case: Training mode completion');
  console.log(`Current question index: ${currentQuestionIndex}`);

  // Simulate handleCompleteQuestion
  const nextQuestionIndex = currentQuestionIndex + 1;
  if (nextQuestionIndex < questions.length) {
    console.log(`✅ Immediately moving to question ${nextQuestionIndex + 1}`);
    console.log(`📝 Question: ${questions[nextQuestionIndex].text}`);

    currentQuestionIndex = nextQuestionIndex;
    console.log(`🔄 Updated currentQuestionIndex to: ${currentQuestionIndex}`);
  }
}

// Run tests
testAutoProgression();
testTrainingMode();

// Final state
setTimeout(() => {
  console.log('\n📊 Final Test Results:');
  console.log(`Current question index: ${currentQuestionIndex}`);
  console.log(`Current question: ${questions[currentQuestionIndex]?.text || 'N/A'}`);
  console.log(`Progress: ${currentQuestionIndex + 1}/${questions.length}`);

  console.log('\n✅ Expected behaviors:');
  console.log('1. Normal mode: Auto-progress after 1.5s delay');
  console.log('2. Training mode: Immediate progression');
  console.log('3. End of questions: Completion message');
}, 2000);
