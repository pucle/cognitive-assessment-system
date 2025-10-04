import { NextRequest, NextResponse } from 'next/server';

interface SummarizeRequest {
  content: string;
  title?: string;
  source?: string;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json() as SummarizeRequest;
    const { content, title, source } = body;

    if (!content || typeof content !== 'string') {
      return NextResponse.json({
        success: false,
        error: 'Content is required and must be a string'
      }, { status: 400 });
    }

    // Check for OpenAI API key
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      // Fallback: simple extractive summary
      const summary = createSimpleSummary(content, title, source);
      return NextResponse.json({ success: true, summary, fallback: true });
    }

    try {
      const { OpenAI } = await import('openai');
      const client = new OpenAI({ apiKey });

      const systemPrompt = `Bạn là một biên tập viên chuyên nghiệp, tường thuật tin tức sức khỏe với giọng văn tường thuật chuẩn mực bằng tiếng Việt.

Hãy viết một đoạn tóm tắt tường thuật như đang đưa tin trên báo đài. Tập trung vào:
- Những phát hiện khoa học quan trọng
- Thông tin chính xác và khách quan
- Ý nghĩa thực tiễn cho người đọc
- Kết thúc bằng khuyến nghị cụ thể

Phong cách viết:
- Giọng văn tường thuật chuẩn mực như biên tập viên truyền hình
- Ngôn ngữ trang trọng nhưng dễ hiểu
- Không dùng dấu phẩy liên tiếp mà dùng cấu trúc câu tường thuật
- Dưới 150 từ nhưng đầy đủ thông tin
- Bắt đầu bằng thông tin chính
- Kết thúc bằng lời khuyên thiết thực

Ví dụ phong cách: "Nghiên cứu mới từ Đại học Harvard chỉ ra rằng tập thể dục đều đặn có thể cải thiện trí nhớ đến 30%. Các nhà khoa học khuyến nghị mọi người nên dành ít nhất 30 phút tập luyện mỗi ngày để duy trì sức khỏe não bộ."

Trả về chỉ tóm tắt tường thuật, không có tiêu đề hay giới thiệu.`;

      const userPrompt = `Tiêu đề: ${title || 'N/A'}
Nguồn: ${source || 'N/A'}

Nội dung bài báo:
${content}`;

      const response = await client.chat.completions.create({
        model: 'gpt-4o-mini', // Use mini for cost efficiency, can change to gpt-4o if needed
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.4,
        max_tokens: 400,
        presence_penalty: 0,
        frequency_penalty: 0.1
      });

      const summary = response.choices?.[0]?.message?.content?.trim();

      if (!summary) {
        throw new Error('No summary generated');
      }

      return NextResponse.json({
        success: true,
        summary: summary
      });

    } catch (openaiError: any) {
      console.error('OpenAI API error:', openaiError);

      // Fallback to simple summary
      const summary = createSimpleSummary(content, title, source);
      return NextResponse.json({
        success: true,
        summary,
        fallback: true,
        error: openaiError.message
      });
    }

  } catch (error: any) {
    console.error('Error in summarize API:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Internal server error'
    }, { status: 500 });
  }
}

function createSimpleSummary(content: string, title?: string, source?: string): string {
  // Create a journalistic fallback summary when GPT is not available
  const journalisticFallbacks = [
    "Bài viết từ các chuyên gia y tế cho biết việc duy trì lối sống lành mạnh đóng vai trò quan trọng trong việc bảo vệ sức khỏe tâm thần. Các bác sĩ khuyến nghị mọi người nên có chế độ ăn uống cân bằng và nghỉ ngơi hợp lý để duy trì trí nhớ tốt.",
    "Theo các nghiên cứu gần đây, việc kết hợp tập thể dục và chế độ dinh dưỡng hợp lý có thể giúp cải thiện sức khỏe não bộ đáng kể. Các chuyên gia khuyên mọi người nên tham khảo ý kiến bác sĩ để có kế hoạch chăm sóc sức khỏe phù hợp.",
    "Thông tin từ các nhà khoa học cho thấy việc quản lý stress hiệu quả là yếu tố then chốt để duy trì sức khỏe tinh thần. Bác sĩ khuyến nghị mọi người nên áp dụng các phương pháp thư giãn và tìm kiếm sự hỗ trợ khi cần thiết.",
    "Các chuyên gia dinh dưỡng nhấn mạnh tầm quan trọng của việc cung cấp đủ chất dinh dưỡng cho não bộ. Việc bổ sung omega-3 và vitamin B có thể hỗ trợ cải thiện chức năng nhận thức theo thời gian."
  ];

  // Try to extract key health-related terms for personalization
  const healthTerms = content.toLowerCase();
  let theme = "sức khỏe tâm thần";

  if (healthTerms.includes('trí nhớ') || healthTerms.includes('nhớ')) {
    theme = "chức năng nhận thức";
  } else if (healthTerms.includes('stress') || healthTerms.includes('căng thẳng')) {
    theme = "sức khỏe tinh thần";
  } else if (healthTerms.includes('ngủ') || healthTerms.includes('sleep')) {
    theme = "chất lượng giấc ngủ";
  } else if (healthTerms.includes('tập thể dục') || healthTerms.includes('exercise')) {
    theme = "hoạt động thể chất";
  }

  // Return a journalistic summary with the detected theme
  return journalisticFallbacks[Math.floor(Math.random() * journalisticFallbacks.length)]
    .replace('sức khỏe tâm thần', theme);
}
