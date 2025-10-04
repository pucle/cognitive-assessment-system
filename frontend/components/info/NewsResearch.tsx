"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  BookOpen,
  ExternalLink,
  Calendar,
  Users,
  Filter,
  Search,
  Newspaper,
  Star,
  Clock
} from "lucide-react";
import { useState, useEffect } from "react";
import { researchPapers, type ResearchPaper } from "@/lib/research-papers";

type FilterType = 'all' | 'research' | 'technology'
type UINewsPaper = ResearchPaper & { category?: string; relevance?: string; journal?: string }
const filterOptions: { value: FilterType; label: string }[] = [
  { value: 'all', label: 'Tất cả' },
  { value: 'research', label: 'Nghiên cứu' },
  { value: 'technology', label: 'Công nghệ' },
]

type LiveItem = {
  id: string;
  title: string;
  url: string;
  source: string;
  publishedAt?: string;
  abstract?: string;
  category?: string;
};

export default function NewsResearch() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredPapers, setFilteredPapers] = useState<UINewsPaper[]>(researchPapers as UINewsPaper[]);
  const [filteredLiveItems, setFilteredLiveItems] = useState<LiveItem[]>([]);
  const [liveItems, setLiveItems] = useState<LiveItem[]>([]);
  const [isLoadingLive, setIsLoadingLive] = useState(false);

  useEffect(() => {
    let filtered = researchPapers as UINewsPaper[];

    // Filter by category
    if (selectedCategory !== 'all') {
      filtered = filtered.filter((paper: UINewsPaper) => paper.category === selectedCategory);
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter((paper: UINewsPaper) =>
        paper.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        paper.abstract.toLowerCase().includes(searchTerm.toLowerCase()) ||
        paper.authors.some((author: string) => author.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    setFilteredPapers(filtered);
  }, [selectedCategory, searchTerm]);

  useEffect(() => {
    let filtered = liveItems;

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(item =>
        item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.abstract?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.source.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredLiveItems(filtered);
  }, [searchTerm, liveItems]);

  // Fetch live RSS items periodically
  useEffect(() => {
    let cancelled = false;
    const fetchFeeds = async () => {
      try {
        setIsLoadingLive(true);
        const res = await fetch('/api/news/feeds', { cache: 'no-store' });
        const data = await res.json();
        if (!cancelled && data?.success && Array.isArray(data.items)) {
          setLiveItems(data.items.slice(0, 12));
        }
      } catch {}
      finally {
        if (!cancelled) setIsLoadingLive(false);
      }
    };
    fetchFeeds();
    const id = setInterval(fetchFeeds, 1000 * 60 * 10); // refresh every 10 minutes
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'research':
        return 'bg-blue-100 text-blue-800';
      case 'treatment':
        return 'bg-green-100 text-green-800';
      case 'technology':
        return 'bg-purple-100 text-purple-800';
      case 'vietnamese':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getRelevanceIcon = (relevance: string) => {
    switch (relevance) {
      case 'high':
        return <Star className="w-4 h-4 text-yellow-500 fill-current" />;
      case 'medium':
        return <Star className="w-4 h-4 text-gray-400" />;
      case 'low':
        return <Star className="w-4 h-4 text-gray-300" />;
      default:
        return null;
    }
  };

  return (
    <section className="py-20 bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Badge  className="mb-4 bg-indigo-100 text-indigo-800">
            Tin tức & Nghiên cứu
          </Badge>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Tin tức & Nghiên cứu về AI và sa sút trí tuệ
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Theo dõi tin tức mới nhất và các công trình nghiên cứu tiên tiến trong lĩnh vực phân tích giọng nói và phát hiện sớm sa sút trí tuệ
          </p>
        </motion.div>

        {/* Filters and Search */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <Card className="p-6">
            <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
              {/* Search */}
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Tìm kiếm tin tức & nghiên cứu..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Category Filters */}
              <div className="flex flex-wrap gap-2">
                <Filter className="w-4 h-4 text-gray-400 mr-2 self-center" />
                {filterOptions.map((filter: { value: FilterType; label: string }) => (
                  <Button
                    key={filter.value}
                    variant={selectedCategory === filter.value ? "default" : "secondaryOutline"}
                    size="sm"
                    onClick={() => setSelectedCategory(filter.value)}
                    className="text-xs"
                  >
                    {filter.label}
                  </Button>
                ))}
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Live Updates (RSS) */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <Card className="p-6 border-2 border-indigo-200 bg-white/60">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Newspaper className="w-5 h-5 text-indigo-600" />
                <h3 className="text-xl font-semibold text-gray-900">Tin tức mới nhất từ các tạp chí</h3>
                <Badge className="bg-indigo-50 text-indigo-700">Live</Badge>
              </div>
              <div className="text-sm text-gray-500">
                {isLoadingLive ? 'Đang tải...' : `Hiển thị ${filteredLiveItems.length} bài`}
              </div>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredLiveItems.map((item) => (
                <a key={item.id} href={item.url} target="_blank" rel="noopener noreferrer" className="group">
                  <div className="p-4 rounded-lg border hover:border-indigo-300 bg-white shadow-sm hover:shadow-md transition-all">
                    <div className="flex items-center justify-between mb-2">
                      <Badge className="bg-blue-50 text-blue-700">{item.source}</Badge>
                      {item.publishedAt && (
                        <span className="text-xs text-gray-500">{new Date(item.publishedAt).toLocaleDateString()}</span>
                      )}
                    </div>
                    <div className="text-gray-900 font-semibold group-hover:text-indigo-700 line-clamp-2">
                      {item.title}
                    </div>
                    {item.abstract && (
                      <div className="text-sm text-gray-600 mt-1 line-clamp-2">{item.abstract}</div>
                    )}
                  </div>
                </a>
              ))}
              {!isLoadingLive && filteredLiveItems.length === 0 && liveItems.length > 0 && (
                <div className="col-span-full text-sm text-gray-500">Không tìm thấy tin tức nào phù hợp với từ khóa. Vui lòng thử từ khóa khác.</div>
              )}
              {!isLoadingLive && liveItems.length === 0 && (
                <div className="col-span-full text-sm text-gray-500">Không có dữ liệu mới. Vui lòng thử lại sau.</div>
              )}
            </div>
          </Card>
        </motion.div>

        {/* Research Papers Grid (curated) */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPapers.map((paper, index) => (
              <motion.div
                key={paper.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="p-6 h-full hover:shadow-xl transition-all duration-300 border-2 hover:border-blue-200">
                  <div className="flex items-start justify-between mb-4">
                    <Badge className={getCategoryColor(paper.category || 'research')}>
                      {filterOptions.find((f: { value: FilterType; label: string }) => f.value === (paper.category as FilterType))?.label}
                    </Badge>
                    <div className="flex items-center">
                      {getRelevanceIcon(paper.relevance || 'low')}
                    </div>
                  </div>

                  <h3 className="text-lg font-bold text-gray-900 mb-3 line-clamp-2">
                    {paper.title}
                  </h3>

                  <div className="flex items-center text-sm text-gray-600 mb-3">
                    <Users className="w-4 h-4 mr-2" />
                    <span className="truncate">
                      {paper.authors.join(', ')}
                    </span>
                  </div>

                  <div className="flex items-center text-sm text-gray-600 mb-4">
                    <BookOpen className="w-4 h-4 mr-2" />
                    <span className="truncate">{paper.journal}</span>
                    <span className="mx-2">•</span>
                    <Calendar className="w-4 h-4 mr-1" />
                    <span>{paper.year}</span>
                  </div>

                  <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                    {paper.abstract}
                  </p>

                  <div className="flex flex-wrap gap-1 mb-4">
                    {paper.tags.slice(0, 3).map((tag, tagIndex) => (
                      <Badge key={tagIndex} className="text-xs bg-gray-50">
                        {tag}
                      </Badge>
                    ))}
                    {paper.tags.length > 3 && (
                      <Badge className="text-xs bg-gray-50">
                        +{paper.tags.length - 3}
                      </Badge>
                    )}
                  </div>

                  {paper.url && (
                    <Button variant="primaryOutline" size="sm" className="w-full" asChild>
                      <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-center"
                      >
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Đọc nghiên cứu
                      </a>
                    </Button>
                  )}
                </Card>
              </motion.div>
            ))}
          </div>

          {filteredPapers.length === 0 && (
            <div className="text-center py-12">
              <BookOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-600 mb-2">
                Không tìm thấy nghiên cứu nào
              </h3>
              <p className="text-gray-500">
                Hãy thử điều chỉnh bộ lọc hoặc từ khóa tìm kiếm
              </p>
            </div>
          )}
        </motion.div>

        {/* RSS Feed Info */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <Card className="p-8 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200">
            <div className="text-center">
              <Newspaper className="w-16 h-16 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Cập nhật nghiên cứu tự động
              </h3>
              <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
                Chúng tôi theo dõi các nguồn nghiên cứu uy tín về sa sút trí tuệ và AI để cập nhật thông tin mới nhất
              </p>

              <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600 mb-2">Alzheimer&apos;s Research</div>
                  <div className="text-sm text-gray-600">BioMed Central</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600 mb-2">Alzheimer&apos;s &amp; Dementia</div>
                  <div className="text-sm text-gray-600">Wiley Online</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600 mb-2">Frontiers in Aging</div>
                  <div className="text-sm text-gray-600">Frontiers Journals</div>
                </div>
              </div>

              <div className="mt-6 flex items-center justify-center text-sm text-gray-600">
                <Clock className="w-4 h-4 mr-2" />
                <span>Cập nhật hàng tuần</span>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <Card className="p-8 bg-gradient-to-r from-indigo-50 to-purple-50 border-2 border-indigo-200">
            <BookOpen className="w-16 h-16 text-indigo-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Đóng góp nghiên cứu
            </h3>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              Bạn có nghiên cứu về AI và sa sút trí tuệ? Hãy chia sẻ để cùng nhau phát triển cộng đồng
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="primary" className="shadow-md" asChild>
                <a href="https://mail.google.com/mail/?view=cm&fs=1&to=ledinhphuc1408@gmail.com" target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="w-4 h-4 mr-2 text-white" />
                  Gửi nghiên cứu
                </a>
              </Button>
              <Button variant="primaryOutline" asChild>
                <a href="https://www.facebook.com/fucdin" target="_blank" rel="noopener noreferrer">
                  <Newspaper className="w-4 h-4 mr-2" />
                  Theo dõi cập nhật
                </a>
              </Button>
            </div>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
