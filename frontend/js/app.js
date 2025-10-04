class VietnameseTranscriber {
    constructor() {
        this.apiUrl = 'http://localhost:8000/api';
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.websocket = null;
        this.history = JSON.parse(localStorage.getItem('transcriptionHistory')) || [];
        
        this.initializeElements();
        this.initializeEventListeners();
        this.checkServerStatus();
        this.loadHistory();
    }
    
    initializeElements() {
        // Status elements
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
        this.modelName = document.getElementById('model-name');
        
        // Control elements
        this.audioFile = document.getElementById('audio-file');
        this.recordBtn = document.getElementById('record-btn');
        this.stopBtn = document.getElementById('stop-btn');
        
        // Settings
        this.realtimeMode = document.getElementById('realtime-mode');
        this.noiseReduction = document.getElementById('noise-reduction');
        this.vadEnabled = document.getElementById('vad-enabled');
        
        // Transcript elements
        this.transcriptBox = document.getElementById('transcript-box');
        this.confidenceScore = document.getElementById('confidence-score');
        this.processingTime = document.getElementById('processing-time');
        
        // Action buttons
        this.copyBtn = document.getElementById('copy-btn');
        this.downloadBtn = document.getElementById('download-btn');
        this.clearBtn = document.getElementById('clear-btn');
        
        // History
        this.historyList = document.getElementById('history-list');
    }
    
    initializeEventListeners() {
        // File upload
        this.audioFile.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Recording controls
        this.recordBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        
        // Action buttons
        this.copyBtn.addEventListener('click', () => this.copyTranscript());
        this.downloadBtn.addEventListener('click', () => this.downloadTranscript());
        this.clearBtn.addEventListener('click', () => this.clearTranscript());
        
        // Settings
        this.realtimeMode.addEventListener('change', () => this.toggleRealtimeMode());
    }
    
    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const status = await response.json();
            
            if (response.ok) {
                this.updateStatus('connected', 'Connected');
                this.modelName.textContent = status.model || 'Unknown';
            } else {
                throw new Error('Server error');
            }
        } catch (error) {
            this.updateStatus('error', 'Connection Error');
            this.modelName.textContent = 'N/A';
        }
    }
    
    updateStatus(status, text) {
        const colors = {
            connected: 'bg-green-500',
            recording: 'bg-red-500',
            processing: 'bg-yellow-500',
            error: 'bg-red-500',
            disconnected: 'bg-gray-400'
        };
        
        this.statusIndicator.className = `w-3 h-3 rounded-full ${colors[status] || colors.disconnected}`;
        this.statusText.textContent = text;
        
        if (status === 'recording') {
            this.statusIndicator.classList.add('recording');
        } else {
            this.statusIndicator.classList.remove('recording');
        }
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        this.updateStatus('processing', 'Processing...');
        this.transcriptBox.classList.add('active');
        
        const startTime = Date.now();
        
        try {
            const formData = new FormData();
            formData.append('audio', file);
            
            const response = await fetch(`${this.apiUrl}/transcribe`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            const processingTime = (Date.now() - startTime) / 1000;
            
            this.displayTranscriptionResult(result, processingTime, file.name);
            this.updateStatus('connected', 'Connected');
            
        } catch (error) {
            console.error('Transcription failed:', error);
            this.displayError('Transcription failed. Please try again.');
            this.updateStatus('error', 'Error');
        }
        
        this.transcriptBox.classList.remove('active');
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: this.noiseReduction.checked
                } 
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    
                    // Real-time processing if enabled
                    if (this.realtimeMode.checked && this.websocket) {
                        this.sendAudioChunk(event.data);
                    }
                }
            };
            
            this.mediaRecorder.onstop = () => {
                if (!this.realtimeMode.checked) {
                    this.processRecordedAudio();
                }
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            
            // Update UI
            this.recordBtn.disabled = true;
            this.recordBtn.classList.add('cursor-not-allowed', 'bg-gray-400');
            this.stopBtn.disabled = false;
            this.stopBtn.classList.remove('cursor-not-allowed', 'bg-gray-400');
            this.stopBtn.classList.add('bg-red-500', 'hover:bg-red-600');
            
            this.updateStatus('recording', 'Recording...');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.displayError('Failed to access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            
            // Update UI
            this.recordBtn.disabled = false;
            this.recordBtn.classList.remove('cursor-not-allowed', 'bg-gray-400');
            this.recordBtn.classList.add('bg-red-500', 'hover:bg-red-600');
            this.stopBtn.disabled = true;
            this.stopBtn.classList.add('cursor-not-allowed', 'bg-gray-400');
            this.stopBtn.classList.remove('bg-red-500', 'hover:bg-red-600');
            
            this.updateStatus('processing', 'Processing...');
        }
    }
    
    async processRecordedAudio() {
        if (this.audioChunks.length === 0) return;
        
        const startTime = Date.now();
        
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            const response = await fetch(`${this.apiUrl}/transcribe`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            const processingTime = (Date.now() - startTime) / 1000;
            
            this.displayTranscriptionResult(result, processingTime, 'Recording');
            this.updateStatus('connected', 'Connected');
            
        } catch (error) {
            console.error('Processing failed:', error);
            this.displayError('Failed to process recording.');
            this.updateStatus('error', 'Error');
        }
    }
    
    displayTranscriptionResult(result, processingTime, filename) {
        if (result.success) {
            this.transcriptBox.innerHTML = `
                <div class="space-y-2">
                    <p class="text-gray-800 font-medium">${result.transcript}</p>
                    <div class="text-sm text-gray-500">
                        <span>Confidence: ${(result.confidence * 100).toFixed(1)}%</span>
                        <span class="mx-2">•</span>
                        <span>Processing: ${processingTime.toFixed(2)}s</span>
                    </div>
                </div>
            `;
            
            this.confidenceScore.textContent = `${(result.confidence * 100).toFixed(0)}%`;
            this.processingTime.textContent = `${processingTime.toFixed(2)}s`;
            
            // Add to history
            this.addToHistory({
                transcript: result.transcript,
                confidence: result.confidence,
                processingTime: processingTime,
                filename: filename,
                timestamp: new Date().toISOString()
            });
            
        } else {
            this.displayError(result.error || 'Transcription failed');
        }
    }
    
    displayError(message) {
        this.transcriptBox.innerHTML = `
            <div class="text-center text-red-600">
                <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                <p>${message}</p>
            </div>
        `;
        
        this.confidenceScore.textContent = '0%';
        this.processingTime.textContent = '0.0s';
    }
    
    copyTranscript() {
        const transcript = this.transcriptBox.textContent.trim();
        if (transcript && !transcript.includes('Your transcription will appear here')) {
            navigator.clipboard.writeText(transcript).then(() => {
                this.showNotification('Transcript copied to clipboard!');
            });
        }
    }
    
    downloadTranscript() {
        const transcript = this.transcriptBox.textContent.trim();
        if (transcript && !transcript.includes('Your transcription will appear here')) {
            const blob = new Blob([transcript], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcript_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showNotification('Transcript downloaded!');
        }
    }
    
    clearTranscript() {
        this.transcriptBox.innerHTML = `
            <p class="text-gray-400 text-center">
                <i class="fas fa-comments text-2xl mb-2"></i><br>
                Your transcription will appear here...
            </p>
        `;
        
        this.confidenceScore.textContent = '0%';
        this.processingTime.textContent = '0.0s';
    }
    
    addToHistory(item) {
        this.history.unshift(item);
        if (this.history.length > 10) {
            this.history = this.history.slice(0, 10);
        }
        
        localStorage.setItem('transcriptionHistory', JSON.stringify(this.history));
        this.loadHistory();
    }
    
    loadHistory() {
        if (this.history.length === 0) {
            this.historyList.innerHTML = '<p class="text-gray-400 text-center py-8">No transcriptions yet</p>';
            return;
        }
        
        this.historyList.innerHTML = this.history.map(item => `
            <div class="bg-gray-50 p-4 rounded-lg">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="font-medium text-gray-800">${item.filename}</h3>
                    <span class="text-sm text-gray-500">${new Date(item.timestamp).toLocaleString()}</span>
                </div>
                <p class="text-gray-700 mb-2">${item.transcript}</p>
                <div class="flex space-x-4 text-sm text-gray-500">
                    <span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>
                    <span>Processing: ${item.processingTime.toFixed(2)}s</span>
                </div>
            </div>
        `).join('');
    }
    
    toggleRealtimeMode() {
        if (this.realtimeMode.checked) {
            this.connectWebSocket();
        } else {
            this.disconnectWebSocket();
        }
    }
    
    connectWebSocket() {
        try {
            this.websocket = new WebSocket('ws://localhost:8000/ws/transcribe');
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'transcript') {
                    this.displayTranscriptionResult(data.result, data.processingTime, 'Real-time');
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.realtimeMode.checked = false;
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.realtimeMode.checked = false;
        }
    }
    
    disconnectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
    
    sendAudioChunk(audioData) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(audioData);
        }
    }
    
    showNotification(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg z-50';
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new VietnameseTranscriber();
});
