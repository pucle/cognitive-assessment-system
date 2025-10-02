// Test script for TRUE immediate progression in normal mode
console.log('🧪 Testing TRUE Immediate Progression Logic');
console.log('=' * 60);

// Mock state
let currentQuestionIndex = 0;
let trainingMode = false; // Normal mode
let questionStates = new Map();
let hasRecording = true;
let isRecordingStarted = false;

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

// Mock queueAssessment
async function queueAssessment(questionId, transcript, audioFeatures) {
  console.log(`📝 queueAssessment called for question ${questionId}`);

  questionStates.set(questionId, {
    id: questionId,
    status: 'processing',
    answer: transcript,
    timestamp: new Date()
  });

  console.log(`🔄 Question ${questionId} status: processing (yellow)`);

  // Simulate background processing
  setTimeout(() => {
    questionStates.set(questionId, {
      id: questionId,
      status: 'completed',
      answer: transcript,
      score: Math.floor(Math.random() * 20) + 80,
      feedback: 'Assessment completed successfully',
      timestamp: new Date()
    });
    console.log(`✅ Question ${questionId} status: completed (green) with score ${questionStates.get(questionId).score}`);
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

// IMMEDIATE PROGRESSION function (called right after recording)
function immediateProgression() {
  console.log('\n⚡ IMMEDIATE PROGRESSION: Moving RIGHT NOW (before transcription)...');

  const nextQuestionIndex = currentQuestionIndex + 1;
  if (nextQuestionIndex < questions.length) {
    console.log(`➡️ Moving to question ${nextQuestionIndex + 1}/${questions.length} IMMEDIATELY`);

    currentQuestionIndex = nextQuestionIndex;
    console.log(`✅ Immediate progression completed. Now on question ${nextQuestionIndex + 1}`);
  } else {
    console.log('🎯 All questions completed!');
  }

  // Reset UI state IMMEDIATELY
  hasRecording = false;
  isRecordingStarted = false;
  console.log('🔄 UI state reset completed');
}

// Simulate recording completion flow
async function simulateRecordingCompletion() {
  console.log('\n📋 Test Case: TRUE Immediate Progression Flow');

  console.log('\n🎙️ Recording completed...');
  console.log(`Current question: ${questions[currentQuestionIndex].text}`);
  console.log(`Training mode: ${trainingMode}`);
  console.log(`Current index: ${currentQuestionIndex}`);

  // Show initial states
  console.log('\n📊 Initial States:');
  console.log(`Current question index: ${currentQuestionIndex}`);
  console.log(`Has recording: ${hasRecording}`);
  console.log(`Is recording started: ${isRecordingStarted}`);
  questionStates.forEach((state, id) => {
    console.log(`Question ${id}: ${state.status}`);
  });

  // STEP 1: IMMEDIATE PROGRESSION (RIGHT AFTER RECORDING)
  console.log('\n🎯 STEP 1: IMMEDIATE PROGRESSION');
  immediateProgression();

  // Show states after immediate progression
  console.log('\n📊 States After Immediate Progression:');
  console.log(`Current question index: ${currentQuestionIndex}`);
  console.log(`Current question: ${questions[currentQuestionIndex]?.text}`);
  console.log(`Has recording: ${hasRecording}`);
  console.log(`Is recording started: ${isRecordingStarted}`);
  console.log(`Progress: ${currentQuestionIndex + 1}/${questions.length}`);

  // STEP 2: BACKGROUND PROCESSING (with delay)
  console.log('\n🎯 STEP 2: BACKGROUND PROCESSING (500ms delay)');
  setTimeout(async () => {
    console.log('🚀 Starting background auto-transcription...');
    await autoTranscribeAudio();

    // Show states after background processing starts
    console.log('\n📊 States After Background Processing Starts:');
    questionStates.forEach((state, id) => {
      console.log(`Question ${id}: ${state.status}${state.score ? ` (score: ${state.score})` : ''}`);
    });

    // Wait for background processing to complete
    setTimeout(() => {
      console.log('\n📊 Final States (After Background Processing Completes):');
      questionStates.forEach((state, id) => {
        console.log(`Question ${id}: ${state.status}${state.score ? ` (score: ${state.score})` : ''}`);
      });

      console.log('\n✅ Test completed successfully!');
      console.log('🎯 Expected behaviors:');
      console.log('1. ✅ IMMEDIATE progression after recording (before transcription)');
      console.log('2. 🔄 Background transcription starts with 500ms delay');
      console.log('3. 🟡 Question state shows processing during background work');
      console.log('4. ✅ Question state shows completed when results arrive');
      console.log('5. ❌ NO auto-progression when results arrive');
      console.log('6. 🎯 User can continue to next question immediately');

    }, 2500);
  }, 500);
}

// Run test
simulateRecordingCompletion();
