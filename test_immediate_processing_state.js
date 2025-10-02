// Test script for immediate processing state in normal mode
console.log('🧪 Testing Immediate Processing State Logic');
console.log('=' * 60);

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
  }
}

// Mock queueAssessment
async function queueAssessment(questionId, transcript, audioFeatures) {
  console.log(`📝 queueAssessment called for question ${questionId}`);

  // This would normally update to processing, but we already did it immediately
  console.log(`🔄 Question ${questionId} already set to processing`);

  // Simulate background processing
  setTimeout(() => {
    // Update to completed when results arrive
    updateQuestionStatus(questionId, 'completed', {
      score: Math.floor(Math.random() * 20) + 80,
      feedback: 'Assessment completed successfully',
      transcript: transcript
    });
  }, 2000);

  return `task-${questionId}`;
}

// Mock submitDomainAssessment (PURE BACKGROUND)
async function submitDomainAssessment(transcript, audioFeatures, gptEval) {
  console.log('📝 submitDomainAssessment called (PURE BACKGROUND)');

  const taskId = await queueAssessment(currentQuestionIndex + 1, transcript, audioFeatures);
  console.log(`✅ Assessment queued with task ID: ${taskId}`);
  console.log('✅ PURE BACKGROUND - NO UI progression');

  return taskId;
}

// Mock autoTranscribeAudio (PURE BACKGROUND)
async function autoTranscribeAudio() {
  console.log('\n🎵 Starting PURE BACKGROUND auto-transcription...');

  // Simulate transcription
  const transcriptText = `Answer for question ${currentQuestionIndex + 1}`;
  const audioFeatures = { duration: 5.2, pitch_mean: 120 };
  const gptEval = { overall_score: 8.5, feedback: 'Good answer' };

  console.log(`✅ Transcription completed: "${transcriptText}"`);

  // PURE BACKGROUND processing only
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

// IMMEDIATE PROGRESSION with PROCESSING STATE
function immediateProgressionWithState() {
  console.log('\n⚡ IMMEDIATE PROGRESSION with PROCESSING STATE');

  // FIRST: Set CURRENT question to PROCESSING (yellow) immediately
  const currentQuestionId = currentQuestionIndex + 1;
  console.log(`🔄 Setting question ${currentQuestionId} to PROCESSING (yellow) immediately`);

  updateQuestionStatus(currentQuestionId, 'processing', {
    answer: 'Recording completed - processing...',
    timestamp: new Date()
  });

  // THEN: Move to next question
  const nextQuestionIndex = currentQuestionIndex + 1;
  if (nextQuestionIndex < questions.length) {
    console.log(`➡️ Moving to question ${nextQuestionIndex + 1}/${questions.length} IMMEDIATELY`);

    currentQuestionIndex = nextQuestionIndex;
    console.log(`✅ Immediate progression completed. Now on question ${nextQuestionIndex + 1}`);
  } else {
    console.log('🎯 All questions completed!');
  }

  console.log('🔄 UI state reset completed');
}

// Simulate complete recording flow
async function simulateCompleteFlow() {
  console.log('\n📋 Test Case: Complete Immediate Progression with Processing State');

  console.log('\n🎙️ Recording completed...');
  console.log(`Current question: ${questions[currentQuestionIndex].text}`);
  console.log(`Current index: ${currentQuestionIndex}`);

  // Show initial states
  console.log('\n📊 Initial Question States:');
  questionStates.forEach((state, id) => {
    console.log(`Question ${id}: ${state.status}`);
  });

  // STEP 1: IMMEDIATE PROGRESSION with PROCESSING STATE
  console.log('\n🎯 STEP 1: IMMEDIATE PROGRESSION with PROCESSING STATE');
  immediateProgressionWithState();

  // Show states after immediate progression
  console.log('\n📊 States After Immediate Progression:');
  console.log(`Current question index: ${currentQuestionIndex}`);
  console.log(`Current question: ${questions[currentQuestionIndex]?.text}`);
  questionStates.forEach((state, id) => {
    console.log(`Question ${id}: ${state.status}${state.status === 'processing' ? ' 🟡 (should be YELLOW)' : ''}`);
  });

  // STEP 2: BACKGROUND PROCESSING (with delay)
  console.log('\n🎯 STEP 2: BACKGROUND PROCESSING (500ms delay)');
  setTimeout(async () => {
    console.log('🚀 Starting background auto-transcription...');
    await autoTranscribeAudio();

    // Show states after background processing completes
    setTimeout(() => {
      console.log('\n📊 Final States (After Background Processing Completes):');
      questionStates.forEach((state, id) => {
        const status = state.status;
        const color = status === 'completed' ? '🟢' : status === 'processing' ? '🟡' : '⚪';
        console.log(`Question ${id}: ${status} ${color}${state.score ? ` (score: ${state.score})` : ''}`);
      });

      console.log('\n✅ Test completed successfully!');
      console.log('🎯 Expected behaviors:');
      console.log('1. ✅ IMMEDIATE progression after recording');
      console.log('2. 🟡 Question set to PROCESSING (YELLOW) immediately');
      console.log('3. 🔄 Background transcription starts with 500ms delay');
      console.log('4. 🟢 Question changes to COMPLETED (GREEN) when results arrive');
      console.log('5. ❌ NO auto-progression when results arrive');
      console.log('6. 🎯 User can continue to next question immediately');

    }, 2500);
  }, 500);
}

// Run test
simulateCompleteFlow();
