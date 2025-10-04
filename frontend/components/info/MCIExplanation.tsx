"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  CheckCircle,
  XCircle,
  Minus
} from "lucide-react";

export default function MCIExplanation() {
  const stages = [
    {
      id: 1,
      stage: "Healthy",
      name: "Sức khỏe bình thường",
      description: "Không có dấu hiệu suy giảm nhận thức",
      symptoms: ["Nhớ được các sự kiện hàng ngày", "Tập trung tốt", "Ra quyết định nhanh"],
      color: "bg-green-500",
      percentage: 100
    },
    {
      id: 2,
      stage: "MCI",
      name: "Suy giảm nhận thức nhẹ (MCI)",
      description: "Bắt đầu có dấu hiệu quên nhưng chưa ảnh hưởng nhiều đến cuộc sống",
      symptoms: ["Quên chìa khóa, ví tiền", "Khó tập trung hơn", "Chậm ra quyết định"],
      color: "bg-yellow-500",
      percentage: 70
    },
    {
      id: 3,
      stage: "Mild Dementia",
      name: "Sa sút trí tuệ nhẹ",
      description: "Ảnh hưởng đến một số hoạt động hàng ngày",
      symptoms: ["Quên tên người quen", "Mất phương hướng", "Khó thực hiện công việc phức tạp"],
      color: "bg-orange-500",
      percentage: 50
    },
    {
      id: 4,
      stage: "Moderate Dementia",
      name: "Sa sút trí tuệ trung bình",
      description: "Cần hỗ trợ nhiều hơn trong sinh hoạt",
      symptoms: ["Quên hoàn toàn sự kiện", "Mất phương hướng nghiêm trọng", "Cần giúp đỡ vệ sinh cá nhân"],
      color: "bg-red-500",
      percentage: 30
    },
    {
      id: 5,
      stage: "Severe Dementia",
      name: "Sa sút trí tuệ nặng",
      description: "Phụ thuộc hoàn toàn vào người chăm sóc",
      symptoms: ["Mất khả năng giao tiếp", "Không nhận ra người thân", "Cần chăm sóc 24/7"],
      color: "bg-red-700",
      percentage: 10
    }
  ];

  const comparisonData = [
    {
      aspect: "Nhớ được sự kiện hàng ngày",
      normal: "check",
      mci: "check",
      dementia: "x"
    },
    {
      aspect: "Tự lập trong sinh hoạt",
      normal: "check",
      mci: "check",
      dementia: "minus"
    },
    {
      aspect: "Giao tiếp rõ ràng",
      normal: "check",
      mci: "check",
      dementia: "x"
    },
    {
      aspect: "Ra quyết định nhanh",
      normal: "check",
      mci: "minus",
      dementia: "x"
    },
    {
      aspect: "Nhận biết người quen",
      normal: "check",
      mci: "check",
      dementia: "x"
    },
    {
      aspect: "Định hướng thời gian",
      normal: "check",
      mci: "minus",
      dementia: "x"
    }
  ];

  const vietnamStats = [
    { label: "Hiện tại", value: "1.2-1.7 triệu người", percentage: 85 },
    { label: "Dự báo 2050", value: "55→139 triệu (toàn cầu)", percentage: 100 },
    { label: "Người 60+", value: "5-7% mắc dementia", percentage: 65 },
    { label: "Tỷ lệ phát hiện sớm", value: "<1%", percentage: 15 }
  ];

  return (
    <section className="py-20 bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Badge className="mb-4 bg-orange-100 text-orange-800">
            Hiểu về MCI và Sa sút trí tuệ
          </Badge>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Từ quên chìa khóa đến mất phương hướng
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Sa sút trí tuệ là quá trình diễn biến dần dần, phát hiện sớm giúp can thiệp hiệu quả hơn
          </p>
        </motion.div>

        {/* Statistics Cards */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16"
        >
          {vietnamStats.map((stat, index) => (
            <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow">
              <div className="text-2xl font-bold text-gray-900 mb-2">{stat.value}</div>
              <div className="text-gray-600 mb-4">{stat.label}</div>
              <Progress value={stat.percentage} className="h-2" />
            </Card>
          ))}
        </motion.div>

        {/* Interactive Timeline */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <h3 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Các giai đoạn của sa sút trí tuệ
          </h3>

          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-1/2 transform -translate-x-1/2 w-1 h-full bg-gradient-to-b from-green-400 via-yellow-400 via-orange-400 to-red-500 rounded-full hidden md:block"></div>

            <div className="space-y-12">
              {stages.map((stage, index) => (
                <motion.div
                  key={stage.id}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -100 : 100 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: index * 0.2 }}
                  viewport={{ once: true }}
                  className={`flex flex-col md:flex-row items-center ${index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'}`}
                >
                  {/* Content Card */}
                  <div className={`w-full md:w-5/12 ${index % 2 === 0 ? 'md:pr-8' : 'md:pl-8'}`}>
                    <Card className="p-6 hover:shadow-xl transition-all duration-300 border-l-4 border-l-current">
                      <div className="flex items-center mb-4">
                        <div className={`w-4 h-4 ${stage.color} rounded-full mr-3`}></div>
                        <h4 className="text-xl font-bold text-gray-900">{stage.name}</h4>
                      </div>

                      <p className="text-gray-600 mb-4">{stage.description}</p>

                      <div className="space-y-2">
                        {stage.symptoms.map((symptom, symptomIndex) => (
                          <div key={symptomIndex} className="flex items-center text-sm text-gray-700">
                            <CheckCircle className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                            {symptom}
                          </div>
                        ))}
                      </div>

                      {/* Progress Bar */}
                      <div className="mt-4">
                        <div className="flex justify-between text-sm text-gray-600 mb-2">
                          <span>Khả năng nhận thức</span>
                          <span>{stage.percentage}%</span>
                        </div>
                        <Progress value={stage.percentage} className="h-2" />
                      </div>
                    </Card>
                  </div>

                  {/* Timeline Dot */}
                  <div className="hidden md:flex w-2/12 justify-center">
                    <motion.div
                      whileHover={{ scale: 1.2 }}
                      className={`w-6 h-6 ${stage.color} rounded-full border-4 border-white shadow-lg cursor-pointer`}
                    />
                  </div>

                  {/* Empty Space */}
                  <div className="w-full md:w-5/12"></div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Comparison Table */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center text-gray-900 mb-12">
            So sánh lão hóa bình thường vs MCI vs Dementia
          </h3>

          <Card className="overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Khía cạnh nhận thức</th>
                    <th className="px-6 py-4 text-center text-sm font-semibold text-green-700">Lão hóa bình thường</th>
                    <th className="px-6 py-4 text-center text-sm font-semibold text-yellow-700">MCI</th>
                    <th className="px-6 py-4 text-center text-sm font-semibold text-red-700">Dementia</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {comparisonData.map((row, index) => (
                    <motion.tr
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                      viewport={{ once: true }}
                      className="hover:bg-gray-50"
                    >
                      <td className="px-6 py-4 text-sm text-gray-900 font-medium">{row.aspect}</td>
                      <td className="px-6 py-4 text-center">
                        {row.normal === 'check' && <CheckCircle className="w-5 h-5 text-green-500 mx-auto" />}
                        {row.normal === 'minus' && <Minus className="w-5 h-5 text-yellow-500 mx-auto" />}
                        {row.normal === 'x' && <XCircle className="w-5 h-5 text-red-500 mx-auto" />}
                      </td>
                      <td className="px-6 py-4 text-center">
                        {row.mci === 'check' && <CheckCircle className="w-5 h-5 text-green-500 mx-auto" />}
                        {row.mci === 'minus' && <Minus className="w-5 h-5 text-yellow-500 mx-auto" />}
                        {row.mci === 'x' && <XCircle className="w-5 h-5 text-red-500 mx-auto" />}
                      </td>
                      <td className="px-6 py-4 text-center">
                        {row.dementia === 'check' && <CheckCircle className="w-5 h-5 text-green-500 mx-auto" />}
                        {row.dementia === 'minus' && <Minus className="w-5 h-5 text-yellow-500 mx-auto" />}
                        {row.dementia === 'x' && <XCircle className="w-5 h-5 text-red-500 mx-auto" />}
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="mt-6 text-center text-sm text-gray-600">
            <p>💡 <strong>Lưu ý:</strong> MCI có thể chuyển sang dementia hoặc quay về bình thường tùy trường hợp</p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
