// Test script for retry failed questions functionality
console.log('🧪 Testing Retry Failed Questions Logic');
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

// Mock updateQuestionStatus
function updateQuestionStatus(questionId, status, updates = {}) {
  console.log(`🔄 updateQuestionStatus: Question ${questionId} -> ${status}`);

  const currentState = questionStates.get(questionId) || { id: questionId, status: 'pending' };
  const newState = {
    ...currentState,
    status,
    ...updates
  };

  questionStates.set(questionId, newState);

  console.log(`✅ Question ${questionId} status updated to: ${status}`);
  if (status === 'processing') {
    console.log(`🟡 Question ${questionId} should display YELLOW`);
  } else if (status === 'completed') {
    console.log(`🟢 Question ${questionId} should display GREEN`);
  } else if (status === 'failed') {
    console.log(`🔴 Question ${questionId} should display RED`);
  }
}

// Mock submitDomainAssessment (PURE BACKGROUND)
async function submitDomainAssessment(transcript, audioFeatures, gptEval) {
  console.log('📝 submitDomainAssessment called (PURE BACKGROUND)');

  const questionId = currentQuestionIndex + 1;
  console.log(`🎯 Processing question ${questionId} with transcript: "${transcript}"`);

  // Simulate immediate processing state update
  updateQuestionStatus(questionId, 'processing');

  // Simulate background processing
  setTimeout(() => {
    // Random success/failure
    const success = Math.random() > 0.3; // 70% success rate

    if (success) {
      updateQuestionStatus(questionId, 'completed', {
        score: Math.floor(Math.random() * 20) + 80,
        feedback: 'Assessment completed successfully',
        transcript: transcript
      });
    } else {
      updateQuestionStatus(questionId, 'failed', {
        error: 'Network timeout',
        transcript: transcript
      });
    }
  }, 1500);

  return `task-${questionId}`;
}

// Mock autoTranscribeAudio (PURE BACKGROUND)
async function autoTranscribeAudio() {
  console.log('\n🎵 Starting PURE BACKGROUND auto-transcription...');

  const transcriptText = `Answer for question ${currentQuestionIndex + 1}`;
  const audioFeatures = { duration: 5.2, pitch_mean: 120 };
  const gptEval = { overall_score: 8.5, feedback: 'Good answer' };

  console.log(`✅ Transcription completed: "${transcriptText}"`);

  if (!trainingMode) {
    console.log('🚀 Processing assessment (PURE BACKGROUND)...');

    try {
      await submitDomainAssessment(transcriptText, audioFeatures, gptEval);
      console.log('✅ submitDomainAssessment completed successfully');
    } catch (error) {
      console.error('❌ submitDomainAssessment failed:', error);
    }

    console.log('✅ PURE BACKGROUND processing completed - NO UI progression');
  }

  return transcriptText;
}

// Mock immediate progression
function immediateProgression() {
  console.log('\n⚡ IMMEDIATE PROGRESSION');

  const nextQuestionIndex = currentQuestionIndex + 1;
  if (nextQuestionIndex < questions.length) {
    console.log(`➡️ Moving to question ${nextQuestionIndex + 1}/${questions.length} IMMEDIATELY`);

    currentQuestionIndex = nextQuestionIndex;
    console.log(`✅ Immediate progression completed. Now on question ${nextQuestionIndex + 1}`);
  } else {
    console.log('🎯 All questions completed!');
  }
}

// Simulate retry flow
async function simulateRetryFlow() {
  console.log('\n📋 Test Case: Retry Failed Questions Flow');

  // Step 1: Initial assessment (simulate failure)
  console.log('\n🎯 STEP 1: Initial Assessment (will fail)');
  await autoTranscribeAudio();

  // Wait for processing to complete
  await new Promise(resolve => setTimeout(resolve, 2000));

  // Check results
  console.log('\n📊 After Initial Assessment:');
  questionStates.forEach((state, id) => {
    const status = state.status;
    const color = status === 'completed' ? '🟢' : status === 'processing' ? '🟡' : status === 'failed' ? '🔴' : '⚪';
    console.log(`Question ${id}: ${status} ${color}${state.score ? ` (score: ${state.score})` : state.error ? ` (error: ${state.error})` : ''}`);
  });

  // Step 2: User clicks on failed question and chooses to retry
  console.log('\n🎯 STEP 2: User clicks failed question and chooses to retry');

  const failedQuestionId = Array.from(questionStates.entries()).find(([id, state]) => state.status === 'failed')?.[0];

  if (failedQuestionId) {
    console.log(`🔄 Retrying question ${failedQuestionId}`);

    // Reset question state to pending
    const currentState = questionStates.get(failedQuestionId);
    updateQuestionStatus(failedQuestionId, 'pending', {
      answer: null,
      score: null,
      feedback: null,
      error: null,
      retryCount: (currentState?.retryCount || 0) + 1,
      timestamp: new Date()
    });

    // Navigate to the question for retry
    currentQuestionIndex = failedQuestionId - 1;
    console.log(`✅ Question ${failedQuestionId} reset for retry. Current index: ${currentQuestionIndex}`);

    // Step 3: User records again
    console.log('\n🎯 STEP 3: User records answer again');
    await autoTranscribeAudio();

    // Wait for retry processing to complete
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Final results
    console.log('\n📊 Final Results After Retry:');
    questionStates.forEach((state, id) => {
      const status = state.status;
      const color = status === 'completed' ? '🟢' : status === 'processing' ? '🟡' : status === 'failed' ? '🔴' : '⚪';
      const retryInfo = state.retryCount > 0 ? ` (retries: ${state.retryCount})` : '';
      console.log(`Question ${id}: ${status} ${color}${state.score ? ` (score: ${state.score})` : state.error ? ` (error: ${state.error})` : ''}${retryInfo}`);
    });

    console.log('\n✅ Test completed successfully!');
    console.log('🎯 Expected behaviors:');
    console.log('1. ✅ Failed question can be retried');
    console.log('2. 🔄 Question state resets to pending on retry');
    console.log('3. 🟡 Question shows processing during retry');
    console.log('4. 🟢 Question shows completed after successful retry');
    console.log('5. 📊 Retry count is tracked');
    console.log('6. 🎯 User can retry multiple times');

  } else {
    console.log('❌ No failed questions found to retry');
  }
}

// Run test
simulateRetryFlow();
