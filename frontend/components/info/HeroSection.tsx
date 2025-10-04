"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { ChevronDown, Brain, TrendingUp, Users } from "lucide-react";

export default function HeroSection() {
  const scrollToAbout = () => {
    const aboutSection = document.getElementById('about-project');
    aboutSection?.scrollIntoView({ behavior: 'smooth' });
  };

  const stats = [
    {
      icon: Users,
      value: "1.2-1.7 triệu",
      label: "Người Việt mắc dementia",
      color: "from-blue-500 to-blue-600"
    },
    {
      icon: TrendingUp,
      value: "<1%",
      label: "Tỷ lệ phát hiện sớm",
      color: "from-red-500 to-red-600"
    },
    {
      icon: Brain,
      value: "AI tiên tiến",
      label: "Phân tích giọng nói",
      color: "from-green-500 to-green-600"
    }
  ];

  // Generate deterministic particle positions/timings to avoid SSR/CSR hydration mismatch
  const particles = useMemo(() => {
    const seeded = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    return Array.from({ length: 20 }, (_, i) => {
      const left = seeded(i * 2.1) * 100;
      const top = seeded(i * 3.7) * 100;
      const duration = 3 + seeded(i * 5.3) * 2;
      const delay = seeded(i * 7.9) * 2;
      // Format with fixed precision to avoid SSR/CSR rounding differences
      return { left: `${left.toFixed(4)}%`, top: `${top.toFixed(4)}%`, duration, delay };
    });
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-blue-700 to-blue-900">
        {/* Animated wave patterns */}
        <div className="absolute inset-0 opacity-10">
          <svg className="w-full h-full" viewBox="0 0 1200 800" preserveAspectRatio="none">
            <defs>
              <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="rgba(255,255,255,0.1)" />
                <stop offset="100%" stopColor="rgba(255,255,255,0.05)" />
              </linearGradient>
            </defs>
            <path d="M0,400 Q300,300 600,400 T1200,400 L1200,800 L0,800 Z" fill="url(#waveGradient)">
              <animate attributeName="d" dur="8s" repeatCount="indefinite"
                values="M0,400 Q300,300 600,400 T1200,400 L1200,800 L0,800 Z;
                        M0,400 Q300,500 600,300 T1200,400 L1200,800 L0,800 Z;
                        M0,400 Q300,350 600,450 T1200,400 L1200,800 L0,800 Z;
                        M0,400 Q300,300 600,400 T1200,400 L1200,800 L0,800 Z"/>
            </path>
          </svg>
        </div>

        {/* Floating particles */}
        <div className="absolute inset-0">
          {particles.map((p, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-white/20 rounded-full"
              style={{ left: p.left as string, top: p.top as string }}
              animate={{ y: [-20, 20, -20], opacity: [0.2, 0.8, 0.2] }}
              transition={{ duration: p.duration, repeat: Infinity, delay: p.delay }}
            />
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-6xl mx-auto px-4 text-center text-white">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-8"
        >
          {/* Brain/Audio Wave Animation */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex justify-center"
          >
            <div className="relative">
              <Brain className="w-20 h-20 text-yellow-300" />
              {/* Animated sound waves */}
              <motion.div
                className="absolute inset-0 flex items-center justify-center"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <div className="w-32 h-32 border-2 border-yellow-300/30 rounded-full"></div>
              </motion.div>
              <motion.div
                className="absolute inset-0 flex items-center justify-center"
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 2.5, repeat: Infinity, delay: 0.5 }}
              >
                <div className="w-40 h-40 border border-yellow-300/20 rounded-full"></div>
              </motion.div>
            </div>
          </motion.div>

          {/* Title */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="space-y-4"
          >
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold leading-tight">
              <span className="text-yellow-300">Cá Vàng</span>
              <br />
              <span className="text-2xl md:text-4xl lg:text-5xl font-medium text-blue-100">
                Thắp Sáng Ký Ức
              </span>
            </h1>
          </motion.div>

          {/* Subtitle */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="max-w-3xl mx-auto"
          >
            <p className="text-xl md:text-2xl text-blue-100 leading-relaxed mb-2">
              Hệ thống AI đa phương thức phân tích giọng nói để sàng lọc sa sút trí tuệ tại nhà
            </p>
            <p className="text-lg text-blue-200">
              Phát triển bởi Lê Đình Phúc và Phan Nguyễn Trà Ly - THPT Chuyên Lê Quý Đôn Đà Nẵng
            </p>
          </motion.div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
          >
            <Button
              onClick={scrollToAbout}
              size="lg"
              className="bg-yellow-500 hover:bg-yellow-600 text-blue-900 font-semibold px-8 py-4 text-lg rounded-full shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
            >
              Khám phá ngay
              <ChevronDown className="ml-2 w-5 h-5" />
            </Button>
          </motion.div>
        </motion.div>

        {/* Key Stats */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.1 }}
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              whileHover={{ scale: 1.05 }}
              className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 shadow-xl"
            >
              <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${stat.color} flex items-center justify-center mx-auto mb-4`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <div className="text-2xl font-bold mb-2">{stat.value}</div>
              <div className="text-blue-100 text-sm">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 2 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center"
        >
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-1 h-3 bg-white/70 rounded-full mt-2"
          />
        </motion.div>
      </motion.div>
    </section>
  );
}
