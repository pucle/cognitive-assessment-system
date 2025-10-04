import { NextRequest, NextResponse } from 'next/server';

type FeedItem = {
  id: string;
  title: string;
  url: string;
  source: string;
  publishedAt?: string;
  abstract?: string;
  category?: string;
};

const RSS_SOURCES: { name: string; url: string; category?: string }[] = [
  { name: 'Alzheimer\'s Research & Therapy', url: 'https://alzres.biomedcentral.com/articles/most-recent/rss', category: 'research' },
  { name: 'Alzheimer\'s & Dementia (Wiley)', url: 'https://alz-journals.onlinelibrary.wiley.com/feed/15525279/most-recent', category: 'research' },
  { name: 'Frontiers in Aging Neuroscience', url: 'https://www.frontiersin.org/journals/aging-neuroscience/rss', category: 'research' },
  // Vietnamese outlets (health/science)
  { name: 'VnExpress - Sức khỏe', url: 'https://vnexpress.net/rss/suc-khoe.rss', category: 'vietnamese' },
  { name: 'VnExpress - Khoa học', url: 'https://vnexpress.net/rss/khoa-hoc.rss', category: 'vietnamese' },
  { name: 'Tuổi Trẻ - Sức khỏe', url: 'https://tuoitre.vn/rss/suc-khoe.rss', category: 'vietnamese' },
  { name: 'Dân Trí - Sức khỏe', url: 'https://dantri.com.vn/suc-khoe.rss', category: 'vietnamese' },
  { name: 'Vietnamnet - Sức khỏe', url: 'https://vietnamnet.vn/rss/suc-khoe.rss', category: 'vietnamese' },
  // International health/science
  { name: 'WHO - News', url: 'https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml', category: 'research' },
  { name: 'NIH News in Health', url: 'https://newsinhealth.nih.gov/rss/all.xml', category: 'research' },
  { name: 'Nature - Neuroscience', url: 'https://www.nature.com/subjects/neuroscience.rss', category: 'research' },
];

function textBetween(xml: string, tag: string): string | undefined {
  const match = xml.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i'));
  return match ? match[1].trim() : undefined;
}

function stripTags(html?: string): string | undefined {
  if (!html) return html;
  return html.replace(/<[^>]*>/g, '').replace(/&nbsp;/g, ' ').trim();
}

function parseRssXml(xml: string, sourceName: string, category?: string, maxItems = 15): FeedItem[] {
  const items: FeedItem[] = [];
  const itemRegex = /<item[\s\S]*?<\/item>/gi;
  const matches = xml.match(itemRegex) || [];
  for (const raw of matches.slice(0, maxItems)) {
    const title = stripTags(textBetween(raw, 'title')) || 'Untitled';
    const link = textBetween(raw, 'link') || textBetween(raw, 'guid') || '';
    const description = stripTags(textBetween(raw, 'description')) || stripTags(textBetween(raw, 'summary'));
    const pubDate = stripTags(textBetween(raw, 'pubDate')) || stripTags(textBetween(raw, 'updated'));
    const id = `${sourceName}:${link || title}:${pubDate || ''}`;
    items.push({
      id,
      title,
      url: link,
      source: sourceName,
      publishedAt: pubDate,
      abstract: description,
      category,
    });
  }
  return items;
}

export async function GET(req: NextRequest) {
  try {
    const results: FeedItem[] = [];

    await Promise.all(
      RSS_SOURCES.map(async (src) => {
        try {
          const res = await fetch(src.url, { next: { revalidate: 600 } });
          if (!res.ok) throw new Error(`Failed to fetch: ${src.url}`);
          const xml = await res.text();
          const items = parseRssXml(xml, src.name, src.category);
          results.push(...items);
        } catch (e) {
          // Continue on single-source failure
        }
      })
    );

    // Memory & Health-related filtering (VN + EN)
    const normalize = (s?: string) => (s || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
    const MEMORY_KEYWORDS = [
      // English
      'memory', 'amnesia', 'hippocamp', 'recall', 'working memory', 'short term memory', 'long term memory', 'mnemonic',
      // Vietnamese (de-accented forms)
      'tri nho', 'suy giam tri nho', 'nho ngan han', 'nho dai han', 'ghi nho'
    ].map(normalize);
    const HEALTH_KEYWORDS = [
      // English
      'health', 'healthcare', 'wellbeing', 'neurology', 'neuroscience', 'brain', 'cognitive', 'cognition', 'dementia', 'alzheimer', 'mci', 'mild cognitive impairment',
      // Vietnamese (de-accented forms)
      'suc khoe', 'y te', 'than kinh', 'nao bo', 'nhan thuc', 'sa sut tri tue', 'tri nao'
    ].map(normalize);

    // Deduplicate by URL (or ID if missing)
    const seen = new Set<string>();
    const deduped = results.filter((item) => {
      const key = item.url || item.id;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    const filtered = deduped.filter((item) => {
      const text = normalize(`${item.title} ${item.abstract || ''}`);
      return MEMORY_KEYWORDS.some((kw) => text.includes(kw)) || HEALTH_KEYWORDS.some((kw) => text.includes(kw));
    });

    // Sort by publishedAt desc when possible
    const sorted = (filtered.length > 0 ? filtered : deduped).sort((a, b) => {
      const ta = a.publishedAt ? Date.parse(a.publishedAt) : 0;
      const tb = b.publishedAt ? Date.parse(b.publishedAt) : 0;
      return tb - ta;
    });

    // If filtering yields nothing, fall back to latest deduped items (limit)
    const limited = (filtered.length > 0 ? sorted : sorted).slice(0, 36);
    return NextResponse.json({ success: true, items: limited });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error?.message || 'Failed to load feeds' }, { status: 500 });
  }
}


