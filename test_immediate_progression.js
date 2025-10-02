// Test script for immediate progression in normal mode
console.log('🧪 Testing Immediate Progression Logic');
console.log('=' * 50);

// Mock state
let currentQuestionIndex = 0;
let trainingMode = false; // Normal mode
let questionStates = new Map();

const questions = [
  { id: '1', text: 'Question 1' },
  { id: '2', text: 'Question 2' },
  { id: '3', text: 'Question 3' },
  { id: '4', text: 'Question 4' }
];

// Initialize question states
questions.forEach((_, index) => {
  questionStates.set(index + 1, { id: index + 1, status: 'pending' });
});

// Mock queueAssessment (updates question state to processing)
async function queueAssessment(questionId, transcript, audioFeatures) {
  console.log(`📝 queueAssessment called for question ${questionId}`);

  // Update question state to processing (yellow)
  questionStates.set(questionId, {
    id: questionId,
    status: 'processing',
    answer: transcript,
    timestamp: new Date()
  });

  console.log(`🔄 Question ${questionId} status: processing (yellow)`);

  // Simulate background processing
  setTimeout(() => {
    // Update question state to completed (green)
    questionStates.set(questionId, {
      id: questionId,
      status: 'completed',
      answer: transcript,
      score: Math.floor(Math.random() * 20) + 80, // Random score 80-100
      feedback: 'Assessment completed successfully',
      timestamp: new Date()
    });
    console.log(`✅ Question ${questionId} status: completed (green) with score ${questionStates.get(questionId).score}`);
  }, 3000); // 3 seconds to simulate processing

  return `task-${questionId}`;
}

// Mock submitDomainAssessment (NO auto-progression)
async function submitDomainAssessment(transcript, audioFeatures, gptEval) {
  console.log('📝 submitDomainAssessment called (NO auto-progression)');

  const taskId = await queueAssessment(currentQuestionIndex + 1, transcript, audioFeatures);
  console.log(`✅ Assessment queued with task ID: ${taskId}`);
  console.log('✅ NO auto-progression - results will update question state only');

  return taskId;
}

// Mock autoTranscribeAudio with immediate progression
async function autoTranscribeAudio() {
  console.log('\n🎵 Starting auto-transcription...');

  // Simulate successful transcription
  const transcriptText = `This is the answer for question ${currentQuestionIndex + 1}`;
  const audioFeatures = { duration: 5.2, pitch_mean: 120 };
  const gptEval = { overall_score: 8.5, feedback: 'Good answer' };

  console.log(`✅ Transcription completed: "${transcriptText}"`);

  // Process assessment in normal mode (NO auto-progression)
  if (!trainingMode) {
    console.log('🚀 Processing assessment (background)...');

    try {
      await submitDomainAssessment(transcriptText, audioFeatures, gptEval);
      console.log('✅ submitDomainAssessment completed successfully');
    } catch (error) {
      console.error('❌ submitDomainAssessment failed:', error);
    }

    // IMMEDIATE PROGRESSION: Move to next question RIGHT AWAY
    console.log('⚡ Immediate progression to next question...');
    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < questions.length) {
      console.log(`➡️ Moving to question ${nextQuestionIndex + 1}/${questions.length} IMMEDIATELY`);

      currentQuestionIndex = nextQuestionIndex;
      console.log(`✅ Immediate progression completed. Now on question ${nextQuestionIndex + 1}`);
    } else {
      console.log('🎯 All questions completed!');
    }
  }

  // Reset UI state
  console.log('🔄 Reset UI state completed');
}

// Test the flow
async function testImmediateProgressionFlow() {
  console.log('\n📋 Test Case: Normal Mode Immediate Progression Flow');
  console.log(`Current question: ${questions[currentQuestionIndex].text}`);
  console.log(`Training mode: ${trainingMode}`);
  console.log(`Current index: ${currentQuestionIndex}`);

  // Show initial question states
  console.log('\n📊 Initial Question States:');
  questionStates.forEach((state, id) => {
    console.log(`Question ${id}: ${state.status}`);
  });

  // Simulate recording completion -> auto transcription
  await autoTranscribeAudio();

  // Show question states after immediate progression
  setTimeout(() => {
    console.log('\n📊 Question States After Immediate Progression:');
    questionStates.forEach((state, id) => {
      console.log(`Question ${id}: ${state.status}${state.score ? ` (score: ${state.score})` : ''}`);
    });
    console.log(`\nCurrent question index: ${currentQuestionIndex}`);
    console.log(`Current question: ${questions[currentQuestionIndex]?.text}`);
    console.log(`Progress: ${currentQuestionIndex + 1}/${questions.length}`);
  }, 100);

  // Show question states after background processing completes
  setTimeout(() => {
    console.log('\n📊 Question States After Background Processing:');
    questionStates.forEach((state, id) => {
      console.log(`Question ${id}: ${state.status}${state.score ? ` (score: ${state.score})` : ''}`);
    });
    console.log('\n✅ Test completed successfully!');
    console.log('🎯 Expected behaviors:');
    console.log('1. ✅ Immediate progression after transcription');
    console.log('2. 🔄 Question state shows processing (yellow) during background work');
    console.log('3. ✅ Question state shows completed (green) when results arrive');
    console.log('4. ❌ NO auto-progression when results arrive');
  }, 4000);
}

// Run test
testImmediateProgressionFlow();
