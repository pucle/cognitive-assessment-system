// Test script for auto-progression in normal mode
console.log('🧪 Testing Auto-Progression Logic');
console.log('=' * 50);

// Mock state
let currentQuestionIndex = 0;
let trainingMode = false; // Normal mode
let isProcessing = false;

const questions = [
  { id: '1', text: 'Question 1' },
  { id: '2', text: 'Question 2' },
  { id: '3', text: 'Question 3' },
  { id: '4', text: 'Question 4' }
];

// Mock submitDomainAssessment
async function submitDomainAssessment(transcript, audioFeatures, gptEval) {
  console.log('📝 submitDomainAssessment called with:', {
    transcript: transcript.substring(0, 30) + '...',
    hasAudioFeatures: !!audioFeatures,
    hasGptEval: !!gptEval,
    questionId: currentQuestionIndex + 1
  });

  isProcessing = true;

  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 1000));

  console.log('✅ Assessment queued successfully');

  // Simulate auto-progression after delay
  setTimeout(() => {
    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < questions.length) {
      console.log(`➡️ Auto-progressing to question ${nextQuestionIndex + 1}/${questions.length}`);
      console.log(`📝 Next question: ${questions[nextQuestionIndex].text}`);

      currentQuestionIndex = nextQuestionIndex;
      console.log(`🔄 Updated currentQuestionIndex to: ${currentQuestionIndex}`);
    } else {
      console.log('🎯 All questions completed!');
    }
    isProcessing = false;
  }, 1500);

  return 'task-123';
}

// Mock autoTranscribeAudio
async function autoTranscribeAudio() {
  console.log('\n🎵 Starting auto-transcription...');

  // Simulate successful transcription
  const transcriptText = `This is the answer for question ${currentQuestionIndex + 1}`;
  const audioFeatures = { duration: 5.2, pitch_mean: 120 };
  const gptEval = { overall_score: 8.5, feedback: 'Good answer' };

  console.log(`✅ Transcription completed: "${transcriptText}"`);

  // Process assessment and auto-progress in normal mode
  if (!trainingMode) {
    console.log('🚀 Processing assessment and auto-progressing...');
    console.log('📝 Calling submitDomainAssessment with:', {
      transcriptText: transcriptText.substring(0, 50) + '...',
      hasAudioFeatures: !!audioFeatures,
      hasGptEval: !!gptEval,
      currentQuestionIndex
    });

    try {
      await submitDomainAssessment(transcriptText, audioFeatures, gptEval);
      console.log('✅ submitDomainAssessment completed successfully');
    } catch (error) {
      console.error('❌ submitDomainAssessment failed:', error);
    }
  }

  console.log('🔄 Reset UI state completed');
}

// Test the flow
async function testNormalModeFlow() {
  console.log('\n📋 Test Case: Normal Mode Auto-Progression Flow');
  console.log(`Current question: ${questions[currentQuestionIndex].text}`);
  console.log(`Training mode: ${trainingMode}`);
  console.log(`Current index: ${currentQuestionIndex}`);

  // Simulate recording completion -> auto transcription
  await autoTranscribeAudio();

  // Wait for auto-progression
  setTimeout(() => {
    console.log('\n📊 State after auto-progression:');
    console.log(`Current question index: ${currentQuestionIndex}`);
    console.log(`Current question: ${questions[currentQuestionIndex]?.text}`);
    console.log(`Progress: ${currentQuestionIndex + 1}/${questions.length}`);
    console.log(`Is processing: ${isProcessing}`);

    if (currentQuestionIndex < questions.length - 1) {
      console.log('\n🔄 Ready for next recording...');
    } else {
      console.log('\n🎉 Assessment completed!');
    }
  }, 3000);
}

// Run test
testNormalModeFlow();
