"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, ExternalLink, Calendar, User } from "lucide-react";
import Link from "next/link";

interface NewsItem {
  id: string;
  title: string;
  url: string;
  source: string;
  publishedAt?: string;
  abstract?: string;
  category?: string;
}

interface NewsDetailProps {
  articleUrl: string;
}

export default function NewsDetail({ articleUrl }: NewsDetailProps) {
  const [article, setArticle] = useState<NewsItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchArticleDetail = async () => {
      try {
        setLoading(true);
        // Try to find the article in our news feed
        const response = await fetch('/api/news/feeds');
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.items) {
            const foundArticle = data.items.find((item: NewsItem) => item.url === articleUrl);
            if (foundArticle) {
              setArticle(foundArticle);
            } else {
              setError('Không tìm thấy bài viết');
            }
          }
        }
      } catch (e) {
        setError('Không thể tải bài viết');
        console.error('Error fetching article:', e);
      } finally {
        setLoading(false);
      }
    };

    fetchArticleDetail();
  }, [articleUrl]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600">Đang tải bài viết...</p>
        </div>
      </div>
    );
  }

  if (error || !article) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 mb-4">
            <ExternalLink className="w-16 h-16 mx-auto" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">
            {error || 'Không tìm thấy bài viết'}
          </h2>
          <p className="text-gray-600 mb-6">
            Bài viết có thể đã bị xóa hoặc không khả dụng.
          </p>
          <Link href="/info/news">
            <Button className="bg-blue-600 hover:bg-blue-700">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Quay lại tin tức
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Link href="/info/news">
            <Button variant="ghost" className="mb-4">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Quay lại tin tức
            </Button>
          </Link>

          <div className="flex items-center gap-2 mb-4">
            <Badge className="bg-blue-100 text-blue-800">
              {article.category === 'vietnamese' ? '🇻🇳 Tin tức Việt Nam' : '🌍 Tin tức Quốc tế'}
            </Badge>
            {article.category === 'research' && (
              <Badge className="bg-purple-100 text-purple-800">
                🔬 Nghiên cứu
              </Badge>
            )}
          </div>
        </motion.div>

        {/* Article Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="p-8 shadow-xl">
            {/* Title */}
            <h1 className="text-3xl font-bold text-gray-900 mb-6 leading-tight">
              {article.title}
            </h1>

            {/* Meta Information */}
            <div className="flex flex-wrap items-center gap-4 mb-8 text-gray-600 border-b pb-6">
              <div className="flex items-center gap-2">
                <User className="w-4 h-4" />
                <span className="font-medium">{article.source}</span>
              </div>
              {article.publishedAt && (
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  <span>{new Date(article.publishedAt).toLocaleDateString('vi-VN', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}</span>
                </div>
              )}
            </div>

            {/* Abstract/Summary */}
            {article.abstract && (
              <div className="mb-8">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Tóm tắt</h2>
                <div className="text-gray-700 leading-relaxed text-lg">
                  {article.abstract}
                </div>
              </div>
            )}

            {/* Read Full Article Button */}
            <div className="flex justify-center pt-8 border-t">
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors shadow-lg hover:shadow-xl"
              >
                <ExternalLink className="w-5 h-5" />
                Đọc bài viết đầy đủ
              </a>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
