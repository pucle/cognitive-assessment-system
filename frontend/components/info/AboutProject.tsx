"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle,
  Clock,
  Users,
  DollarSign,
  CheckCircle,
  Mic,
  Cpu,
  FileText,
  ArrowRight,
  Zap
} from "lucide-react";

export default function AboutProject() {
  const problems = [
    {
      icon: Users,
      title: "Thực trạng tại Việt Nam",
      stats: "5-7% người 60+ mắc dementia",
      description: "1.2-1.7 triệu người Việt đang sống với bệnh sa sút trí tuệ"
    },
    {
      icon: DollarSign,
      title: "Chi phí khám chuyên khoa",
      stats: "500k-2tr VND/lần",
      description: "Khó tiếp cận với người dân ở khu vực nông thôn"
    },
    {
      icon: Clock,
      title: "Thời gian kiểm tra",
      stats: "10-20 phút",
      description: "Quá trình kiểm tra truyền thống tốn nhiều thời gian của cả bệnh nhân và bác sĩ"
    },
    {
      icon: AlertTriangle,
      title: "Nguồn nhân lực",
      stats: "8/10.000 dân",
      description: "Thiếu bác sĩ chuyên khoa thần kinh ở vùng sâu vùng xa"
    }
  ];

  const solutions = [
    {
      icon: Mic,
      title: "Sàng lọc tại nhà",
      description: "Không cần đến bệnh viện, có thể thực hiện mọi lúc mọi nơi"
    },
    {
      icon: Zap,
      title: "Giảm thời gian xuống còn <5 phút",
      description: "Quy trình nhanh chóng, thuận tiện cho người cao tuổi"
    },
    {
      icon: DollarSign,
      title: "Chi phí gần như miễn phí",
      description: "Tiết kiệm hàng triệu đồng cho mỗi lần kiểm tra"
    },
    {
      icon: Cpu,
      title: "Theo dõi liên tục",
      description: "Phát hiện sớm MCI để can thiệp kịp thời"
    }
  ];

  const pipelineSteps = [
    { id: 1, title: "Ghi âm giọng nói", icon: Mic, description: "Thu thập mẫu giọng nói tự nhiên" },
    { id: 2, title: "Chuyển đổi thành văn bản", icon: FileText, description: "Sử dụng công nghệ ASR tiên tiến" },
    { id: 3, title: "Trích xuất đặc trưng", icon: Cpu, description: "Phân tích âm học và ngôn ngữ" },
    { id: 4, title: "AI phân tích", icon: Zap, description: "Mô hình học máy dự đoán" },
    { id: 5, title: "Xuất báo cáo MMSE", icon: FileText, description: "Kết quả chi tiết và khuyến nghị" }
  ];

  return (
    <section id="about-project" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Badge className="mb-4 bg-blue-100 text-blue-800">
            Về dự án Cá Vàng
          </Badge>
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4 md:mb-6">
            Giải pháp công nghệ cho vấn đề sức khỏe cộng đồng
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Áp dụng trí tuệ nhân tạo để biến smartphone thành công cụ sàng lọc sa sút trí tuệ hiệu quả
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8 md:gap-12 items-start">
          {/* Problems Side */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            <div>
              <h3 className="text-2xl font-bold text-red-600 mb-6 flex items-center">
                <AlertTriangle className="w-8 h-8 mr-3" />
                Vấn đề thực tế
              </h3>
              <div className="space-y-6">
                {problems.map((problem, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <Card className="p-4 md:p-6 border-l-4 border-l-red-500 hover:shadow-lg transition-shadow">
                      <div className="flex items-start space-x-4">
                        <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
                          <problem.icon className="w-6 h-6 text-red-600" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 mb-1">{problem.title}</h4>
                          <div className="text-2xl font-bold text-red-600 mb-2">{problem.stats}</div>
                          <p className="text-gray-600 text-sm">{problem.description}</p>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Solutions Side */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            <div>
              <h3 className="text-2xl font-bold text-green-600 mb-6 flex items-center">
                <CheckCircle className="w-8 h-8 mr-3" />
                Giải pháp của Cá Vàng
              </h3>
              <div className="space-y-6">
                {solutions.map((solution, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <Card className="p-4 md:p-6 border-l-4 border-l-green-500 hover:shadow-lg transition-shadow">
                      <div className="flex items-start space-x-4">
                        <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                          <solution.icon className="w-6 h-6 text-green-600" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 mb-2">{solution.title}</h4>
                          <p className="text-gray-600 text-sm">{solution.description}</p>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Interactive Pipeline */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-12 md:mt-20"
        >
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">Luồng hoạt động</h3>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Quy trình từ ghi âm đến báo cáo chỉ trong vài phút, sử dụng công nghệ AI tiên tiến
            </p>
          </div>

          <div className="relative">
            {/* Pipeline Steps */}
            <div className="flex flex-col md:flex-row items-center justify-between space-y-6 md:space-y-0 md:space-x-4">
              {pipelineSteps.map((step, index) => (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="flex flex-col items-center text-center max-w-xs"
                >
                  <div className="relative">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center mb-4 shadow-lg">
                      <step.icon className="w-8 h-8 text-white" />
                    </div>
                    {index < pipelineSteps.length - 1 && (
                      <ArrowRight className="hidden md:block absolute top-6 -right-12 w-8 h-8 text-blue-400" />
                    )}
                  </div>
                  <h4 className="font-semibold text-gray-900 mb-2">{step.title}</h4>
                  <p className="text-gray-600 text-sm">{step.description}</p>
                </motion.div>
              ))}
            </div>

            {/* Animated Flow Line */}
            <motion.div
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              transition={{ duration: 2, delay: 0.5 }}
              viewport={{ once: true }}
              className="hidden md:block absolute top-8 left-20 right-20 h-0.5 bg-gradient-to-r from-blue-400 via-blue-500 to-blue-600 origin-left"
            />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
