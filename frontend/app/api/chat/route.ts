// app/api/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';

// Chat configuration
const CHAT_CONFIG = {
  temperature: 0.7,
  maxTokens: 2048,
  model: 'gpt-3.5-turbo'
};

export async function POST(request: NextRequest) {
  console.log('🤖 OpenAI Chat Request received');
  
  try {
    const { 
      message, 
      model = 'gpt-3.5-turbo',
      systemPrompt,
      chatHistory = [],
      temperature,
      maxTokens 
    } = await request.json();

    console.log('Request params:', { 
      messageLength: message?.length, 
      model,
      historyLength: chatHistory.length,
      messagePreview: message?.substring(0, 100) 
    });

    // Validate input
    if (!message || typeof message !== 'string') {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json({ 
        error: 'OpenAI API key not configured' 
      }, { status: 500 });
    }

    // Prepare messages for OpenAI
    const messages = [];
    
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    
    // Add chat history
    if (chatHistory.length > 0) {
      messages.push(...chatHistory);
    }
    
    // Add current message
    messages.push({ role: 'user', content: message });

    console.log(`🤖 Using model: ${model}`);

    // Generate response using OpenAI
    console.log('🤖 Generating response...');
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: model,
        messages: messages,
        temperature: temperature || CHAT_CONFIG.temperature,
        max_tokens: maxTokens || CHAT_CONFIG.maxTokens
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`OpenAI API error: ${errorData.error?.message || response.statusText}`);
    }

    const result = await response.json();
    const text = result.choices[0]?.message?.content;

    if (!text) {
      throw new Error('No response generated from OpenAI');
    }

    console.log(`🎉 OpenAI Success: Generated ${text.length} characters`);
    
    return NextResponse.json({
      success: true,
      response: text,
      model: model,
      usage: {
        promptTokens: result.usage?.prompt_tokens || 0,
        completionTokens: result.usage?.completion_tokens || 0,
        totalTokens: result.usage?.total_tokens || 0
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ OpenAI API Error:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { 
        error: 'Failed to generate response',
        details: errorMessage,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

// GET endpoint for model info and health check
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const test = searchParams.get('test');
  
  if (test === 'true') {
    // Test OpenAI connectivity
    try {
      if (!process.env.OPENAI_API_KEY) {
        return NextResponse.json({
          status: 'error',
          message: 'OPENAI_API_KEY not configured'
        });
      }

      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: [{ role: 'user', content: 'Hello, this is a test message.' }],
          max_tokens: 50
        })
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.statusText}`);
      }

      const result = await response.json();
      const text = result.choices[0]?.message?.content;
      
      return NextResponse.json({
        status: 'success',
        openaiAvailable: true,
        testResponse: text?.substring(0, 100) + '...',
        model: 'gpt-3.5-turbo'
      });
    } catch (error) {
      return NextResponse.json({
        status: 'error',
        openaiAvailable: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  // Regular GET response
  return NextResponse.json({
    service: 'OpenAI Chat API',
    models: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
    configuration: {
      temperature: CHAT_CONFIG.temperature,
      maxTokens: CHAT_CONFIG.maxTokens,
      model: CHAT_CONFIG.model
    },
    usage: {
      message: 'Text message to send to OpenAI',
      model: 'Model selection: gpt-3.5-turbo, gpt-4, gpt-4-turbo',
      systemPrompt: 'Optional system prompt to guide AI behavior',
      chatHistory: 'Array of previous messages for context',
      temperature: 'Creativity level (0.0-1.0)',
      maxTokens: 'Maximum response length'
    },
    endpoints: {
      'POST /api/chat': 'Send message to OpenAI',
      'GET /api/chat': 'Get API info',
      'GET /api/chat?test=true': 'Test OpenAI connectivity'
    }
  });
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}