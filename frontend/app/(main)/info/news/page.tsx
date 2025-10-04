"use client";

import { useSearchParams } from "next/navigation";
import NewsResearch from "@/components/info/NewsResearch";
import NewsDetail from "@/components/info/NewsDetail";

export default function InfoNewsPage() {
  const searchParams = useSearchParams();
  const url = searchParams.get('url');

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


