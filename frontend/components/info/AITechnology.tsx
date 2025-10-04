"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Mic,
  FileText,
  Cpu,
  Zap,
  Brain,
  BarChart3,
  Play,
  Volume2,
  Activity,
  CheckCircle
} from "lucide-react";

export default function AITechnology() {

  const acousticFeatures = [
    { name: "Tốc độ nói", description: "Số từ/phút", normal: 85, abnormal: 65 },
    { name: "Cao độ (Pitch)", description: "Tần số cơ bản F0", normal: 78, abnormal: 45 },
    { name: "Khoảng ngắt nghỉ", description: "Thời gian im lặng", normal: 82, abnormal: 35 },
    { name: "Độ ổn định cao độ", description: "F0 contour", normal: 90, abnormal: 55 }
  ];

  const linguisticFeatures = [
    { name: "Đa dạng từ vựng (TTR)", description: "Type-Token Ratio", normal: 88, abnormal: 62 },
    { name: "Tính mạch lạc", description: "Semantic coherence", normal: 85, abnormal: 48 },
    { name: "Từ đệm", description: "Filler words (uh, um)", normal: 75, abnormal: 35 },
    { name: "Độ phức tạp câu", description: "Syntactic complexity", normal: 80, abnormal: 52 }
  ];

  const vietnameseFeatures = [
    { name: "Thanh điệu", description: "Tone patterns", importance: "Cao" },
    { name: "F0 contour", description: "Ngữ điệu tự nhiên", importance: "Cao" },
    { name: "Ngắt nghỉ", description: "Ranh giới từ/câu", importance: "Trung bình" },
    { name: "Tốc độ nói", description: "Giọng miền Nam/Bắc", importance: "Thấp" }
  ];

  const modelLayers = [
    {
      name: "Tầng 1: Phát hiện nguy cơ",
      description: "Sàng lọc nhanh để xác định nguy cơ MCI/Dementia",
      accuracy: 94.2,
      features: ["Sensitivity cao", "Xử lý nhanh", "Chi phí thấp"],
      color: "from-blue-500 to-blue-600"
    },
    {
      name: "Tầng 2: Phân loại chi tiết",
      description: "Phân tích chuyên sâu + dự đoán điểm MMSE",
      accuracy: 87.6,
      features: ["MAE 3.0 điểm", "Chi tiết hơn", "Khuyến nghị cụ thể"],
      color: "from-purple-500 to-purple-600"
    }
  ];

  return (
    <section className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Badge className="mb-4 bg-purple-100 text-purple-800">
            Công nghệ AI và Phương pháp
          </Badge>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Giọng nói phản ánh quá trình suy nghĩ
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Kết hợp âm thanh + ngôn ngữ + AI để dự đoán chính xác tình trạng nhận thức
          </p>
        </motion.div>

        {/* Main Concept */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Card className="p-8 max-w-4xl mx-auto bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-100">
            <div className="grid md:grid-cols-3 gap-8 items-center">
              <div className="text-center">
                <Mic className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Âm thanh</h3>
                <p className="text-gray-600">Tốc độ, cao độ, nhịp điệu</p>
              </div>

              <div className="text-center">
                <Brain className="w-16 h-16 text-purple-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Ngôn ngữ</h3>
                <p className="text-gray-600">Từ vựng, mạch lạc, ngữ pháp</p>
              </div>

              <div className="text-center">
                <Cpu className="w-16 h-16 text-green-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">AI phân tích</h3>
                <p className="text-gray-600">Mô hình học máy tiên tiến</p>
              </div>
            </div>

            <div className="mt-8 text-center">
              <Zap className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
              <p className="text-lg font-medium text-gray-800">
                Kết hợp ba yếu tố tạo nên độ chính xác cao trong sàng lọc
              </p>
            </div>
          </Card>
        </motion.div>

        {/* Feature Analysis */}
        <div className="grid lg:grid-cols-2 gap-12 mb-16">
          {/* Acoustic Features */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            viewport={{ once: true }}
          >
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <Volume2 className="w-6 h-6 mr-2 text-blue-600" />
              Đặc trưng âm học
            </h3>

            <div className="space-y-4">
              {acousticFeatures.map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="p-4 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold text-gray-900">{feature.name}</h4>
                      <div className="flex space-x-4 text-sm">
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                          <span className="text-gray-600">Bình thường: {feature.normal}%</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                          <span className="text-gray-600">Bất thường: {feature.abnormal}%</span>
                        </div>
                      </div>
                    </div>
                    <p className="text-gray-600 text-sm mb-3">{feature.description}</p>
                    <div className="flex space-x-2">
                      <Progress value={feature.normal} className="flex-1 h-2" />
                      <Progress value={feature.abnormal} className="flex-1 h-2" />
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Linguistic Features */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <FileText className="w-6 h-6 mr-2 text-purple-600" />
              Đặc trưng ngôn ngữ
            </h3>

            <div className="space-y-4">
              {linguisticFeatures.map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Card className="p-4 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold text-gray-900">{feature.name}</h4>
                      <div className="flex space-x-4 text-sm">
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                          <span className="text-gray-600">Bình thường: {feature.normal}%</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                          <span className="text-gray-600">Bất thường: {feature.abnormal}%</span>
                        </div>
                      </div>
                    </div>
                    <p className="text-gray-600 text-sm mb-3">{feature.description}</p>
                    <div className="flex space-x-2">
                      <Progress value={feature.normal} className="flex-1 h-2" />
                      <Progress value={feature.abnormal} className="flex-1 h-2" />
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Vietnamese-Specific Features */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h3 className="text-2xl font-bold text-center text-gray-900 mb-8">
            Xử lý đặc thù tiếng Việt
          </h3>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {vietnameseFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="p-6 text-center hover:shadow-lg transition-shadow">
                  <Activity className="w-8 h-8 text-blue-600 mx-auto mb-3" />
                  <h4 className="font-semibold text-gray-900 mb-2">{feature.name}</h4>
                  <p className="text-gray-600 text-sm mb-3">{feature.description}</p>
                  <Badge
                    className={
                      feature.importance === 'Cao'
                        ? 'bg-red-100 text-red-800'
                        : feature.importance === 'Trung bình'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }
                  >
                    Độ quan trọng: {feature.importance}
                  </Badge>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Two-Layer Model Architecture */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Mô hình hai tầng thông minh
          </h3>

          <div className="grid md:grid-cols-2 gap-8">
            {modelLayers.map((layer, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                viewport={{ once: true }}
              >
                <Card className="p-8 hover:shadow-xl transition-all duration-300 border-2">
                  <div className={`w-full h-2 bg-gradient-to-r ${layer.color} rounded-full mb-6`}></div>

                  <h4 className="text-xl font-bold text-gray-900 mb-4">{layer.name}</h4>
                  <p className="text-gray-600 mb-6">{layer.description}</p>

                  <div className="mb-6">
                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                      <span>Độ chính xác</span>
                      <span>{layer.accuracy}%</span>
                    </div>
                    <Progress value={layer.accuracy} className="h-3" />
                  </div>

                  <div className="space-y-2">
                    {layer.features.map((feature, featureIndex) => (
                      <div key={featureIndex} className="flex items-center text-sm text-gray-700">
                        <CheckCircle className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                        {feature}
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Audio Demo Section (Placeholder) */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <Card className="p-8 bg-gradient-to-r from-gray-50 to-blue-50 border-2 border-dashed border-blue-200">
            <BarChart3 className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Demo tương tác (Sắp ra mắt)
            </h3>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              Trải nghiệm phân tích giọng nói thời gian thực và xem F0 contour thay đổi
            </p>
            <Button variant="primaryOutline" disabled className="opacity-50 cursor-not-allowed">
              <Play className="w-4 h-4 mr-2" />
              Phát demo
            </Button>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
