import { NextRequest, NextResponse } from 'next/server';

type EvalInput = {
  questionId?: number | string;
  questionText?: string;
  expectedAnswer?: string;
  responseText: string;
  language?: 'vi' | 'en' | string;
  model?: string;
};

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as EvalInput;
    const language = (body.language as 'vi' | 'en') || 'vi';
    const responseText = (body.responseText || '').trim();
    const questionText = (body.questionText || '').trim();
    const expectedAnswer = (body.expectedAnswer || '').trim();

    // Fallback heuristic when no LLM available
    const heuristicEvaluate = () => {
      const len = responseText.split(/\s+/).filter(Boolean).length;
      const hasExpected = expectedAnswer
        ? expectedAnswer
            .toLowerCase()
            .split(/[,\s]+/)
            .filter(Boolean)
            .some((tok) => responseText.toLowerCase().includes(tok))
        : false;

      const correctness = hasExpected ? 0.9 : Math.min(0.2 + len / 50, 0.7);
      const partialCredit = hasExpected ? 0.2 : Math.min(len / 100, 0.2);
      const confidence = Math.min(0.5 + len / 60, 0.9);
      const overallScore = Math.round(((correctness * 0.8 + partialCredit * 0.2) * 10) * 10) / 10; // 0-10
      const clinicalNotes =
        language === 'vi'
          ? (hasExpected
              ? 'Câu trả lời phù hợp nội dung mong đợi, độ dài đủ để đánh giá.'
              : 'Câu trả lời còn ngắn hoặc thiếu thông tin; cân nhắc hỏi lại để làm rõ.')
          : (hasExpected
              ? 'Response aligns with expected content; length is sufficient.'
              : 'Response is brief/incomplete; consider a follow‑up prompt.');

      return { correctness, partialCredit, clinicalNotes, confidence, overallScore };
    };

    // Optional OpenAI evaluation when key is present
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      const evalResult = heuristicEvaluate();
      return NextResponse.json({ success: true, evaluation: evalResult });
    }

    try {
      const { OpenAI } = await import('openai');
      const client = new OpenAI({ apiKey });
      const prompt = `You are a clinical evaluator for MMSE-like tasks.
Return a strict JSON with fields: correctness (0..1), partialCredit (0..1), clinicalNotes (string), confidence (0..1), overallScore (0..10).
Question: ${questionText}
Expected: ${expectedAnswer}
Response: ${responseText}`;

      const resp = await client.chat.completions.create({
        model: body.model || 'gpt-4o-mini',
        temperature: 0.2,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: 'Return only valid JSON as specified.' },
          { role: 'user', content: prompt },
        ],
      });

      const content = resp.choices?.[0]?.message?.content || '{}';
      const parsed = JSON.parse(content);
      // Basic sanitize
      const evaluation = {
        correctness: Math.min(Math.max(Number(parsed.correctness) || 0, 0), 1),
        partialCredit: Math.min(Math.max(Number(parsed.partialCredit) || 0, 0), 1),
        clinicalNotes: String(parsed.clinicalNotes || ''),
        confidence: Math.min(Math.max(Number(parsed.confidence) || 0, 0), 1),
        overallScore: Math.min(Math.max(Number(parsed.overallScore) || 0, 0), 10),
      };

      return NextResponse.json({ success: true, evaluation });
    } catch (e) {
      const evalResult = heuristicEvaluate();
      return NextResponse.json({ success: true, evaluation: evalResult, fallback: true });
    }
  } catch (error) {
    return NextResponse.json({ success: false, error: 'Invalid request' }, { status: 400 });
  }
}


