"use client";

import NewsResearch from "@/components/info/NewsResearch";
import NewsDetail from "@/components/info/NewsDetail";

interface PageProps {
  searchParams: { page?: string; url?: string } | null;
}

export default function InfoNewsPage({ searchParams }: PageProps) {
  const page = searchParams?.page ?? '1';
  const url = searchParams?.url ?? undefined;

  if (url) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
        <NewsDetail articleUrl={url} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      <NewsResearch />
    </div>
  );
}


