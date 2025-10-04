import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const url = searchParams.get('url');

    if (!url) {
      return NextResponse.json({ success: false, error: 'URL is required' }, { status: 400 });
    }

    // Validate URL
    try {
      new URL(url);
    } catch {
      return NextResponse.json({ success: false, error: 'Invalid URL' }, { status: 400 });
    }

    // Fetch the article content
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      },
      next: { revalidate: 300 } // Cache for 5 minutes
    });

    if (!response.ok) {
      return NextResponse.json({
        success: false,
        error: `Failed to fetch article: ${response.status}`
      }, { status: response.status });
    }

    const html = await response.text();

    // Extract text content from HTML (basic extraction)
    const textContent = extractTextFromHtml(html);

    if (!textContent || textContent.length < 100) {
      return NextResponse.json({
        success: false,
        error: 'Could not extract sufficient content from the article'
      }, { status: 422 });
    }

    return NextResponse.json({
      success: true,
      content: textContent,
      url: url
    });

  } catch (error: any) {
    console.error('Error fetching article content:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to fetch article content'
    }, { status: 500 });
  }
}

function extractTextFromHtml(html: string): string {
  // Remove script and style elements
  let text = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
  text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');

  // Remove HTML comments
  text = text.replace(/<!--[\s\S]*?-->/g, '');

  // Remove HTML tags but keep text content
  text = text.replace(/<[^>]+>/g, ' ');

  // Decode HTML entities
  text = text.replace(/&nbsp;/g, ' ');
  text = text.replace(/&amp;/g, '&');
  text = text.replace(/&lt;/g, '<');
  text = text.replace(/&gt;/g, '>');
  text = text.replace(/&quot;/g, '"');
  text = text.replace(/&#39;/g, "'");

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim();

  // Remove common navigation/footer text
  const removePatterns = [
    /đọc tiếp|đọc thêm|xem thêm|chia sẻ|bình luận|tải ứng dụng|quảng cáo/gi,
    /copyright|©|all rights reserved/gi,
    /điều khoản|chính sách|liên hệ|tuyển dụng/gi,
    /facebook|twitter|instagram|youtube|linkedin/gi
  ];

  removePatterns.forEach(pattern => {
    text = text.replace(pattern, '');
  });

  // Clean up again
  text = text.replace(/\s+/g, ' ').trim();

  // Limit to reasonable length (first 2000 characters should contain main content)
  return text.substring(0, 2000);
}
