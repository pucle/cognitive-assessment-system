"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  User,
  GraduationCap,
  Award,
  Users as UsersIcon,
  Mail,
  MapPin,
  Trophy,
  BookOpen
} from "lucide-react";

export default function TeamSection() {
  const teamMembers = [
    {
      name: "Trần Đức Thanh",
      role: "Giáo viên hướng dẫn",
      school: "THPT Chuyên Lê Quý Đôn",
      specialty: "Chuyên môn thiết kế, báo cáo, nghiên cứu",
      avatar: "/api/placeholder/120/120",
      achievements: ["Hướng dẫn nghiên cứu", "Phát triển AI", "Hỗ trợ tinh thần"]
    },
    {
      name: "Lê Đình Phúc",
      role: "Thành viên chính",
      school: "THPT Chuyên Lê Quý Đôn",
      specialty: "AI & Machine Learning",
      avatar: "/api/placeholder/120/120",
      achievements: ["Lập trình viên chính", "Phát triển mô hình", "Nghiên cứu giọng nói"]
    },
    {
      name: "Phan Nguyễn Trà Ly",
      role: "Thành viên",
      school: "THPT Chuyên Lê Quý Đôn",
      specialty: "Data Science",
      avatar: "/api/placeholder/120/120",
      achievements: ["Xử lý dữ liệu", "Phân tích", "Thiết kế giao diện"]
    }
  ];

  const schoolInfo = {
    name: "THPT Chuyên Lê Quý Đôn",
    location: "Thành phố Đà Nẵng, Việt Nam",
    type: "Trường THPT chuyên",
    specialties: ["Toán học", "Vật lý", "Tin học", "Hóa học", "Sinh học", "Ngữ văn", "Lịch sử", "Địa Lý", "Tiếng Anh", "Tiếng Pháp", "Tiếng Nhật"],
    achievements: [
      "Top 10 trường THPT Việt Nam",
      "Đạt giải cao Olympic Quốc tế",
      "Trung tâm đào tạo tài năng"
    ],
  };

  const competitionInfo = {
    name: "Bảng B - THPT",
    description: "Dự án hướng đến sức khỏe cộng đồng",
    year: 2025,
    category: "Phần mềm và Ứng dụng thông minh",
    project: "Cá Vàng: Thắp Sáng Ký Ức"
  };

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
          <Badge className="mb-4 bg-blue-100 text-blue-800">
            Nhóm phát triển
          </Badge>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Đội ngũ đam mê công nghệ
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Học sinh THPT Chuyên Lê Quý Đôn với sứ mệnh áp dụng AI vào chăm sóc sức khỏe cộng đồng
          </p>
        </motion.div>

        {/* Competition Info */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <Card className="p-8 bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-200">
            <div className="text-center">
              <Trophy className="w-16 h-16 text-yellow-600 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                {competitionInfo.name}
              </h3>
              <p className="text-gray-600 mb-6">{competitionInfo.description}</p>

              <div className="grid md:grid-cols-3 gap-6 max-w-3xl mx-auto">
                <div>
                  <div className="text-2xl font-bold text-blue-600 mb-1">{competitionInfo.year}</div>
                  <div className="text-gray-600">Năm tham dự</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-purple-600 mb-1">{competitionInfo.category}</div>
                  <div className="text-gray-600">Lĩnh vực</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-green-600 mb-1">{competitionInfo.project}</div>
                  <div className="text-gray-600">Dự án</div>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Team Members */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <h3 className="text-2xl md:text-3xl font-bold text-center text-gray-900 mb-8 md:mb-12">
            Thành viên đội Cá Vàng
          </h3>

          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4 md:gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                viewport={{ once: true }}
              >
                <Card className="p-4 md:p-6 text-center hover:shadow-xl transition-all duration-300 border">
                  <div className="w-16 h-16 md:w-24 md:h-24 bg-gradient-to-br from-blue-400 to-blue-600 rounded-full mx-auto mb-3 md:mb-4 flex items-center justify-center">
                    <User className="w-8 h-8 md:w-12 md:h-12 text-white" />
                  </div>

                  <h4 className="text-lg md:text-xl font-bold text-gray-900 mb-1">{member.name}</h4>
                  <Badge  className="mb-2 md:mb-3 bg-blue-50 text-blue-700">
                    {member.role}
                  </Badge>

                  <div className="space-y-2 mb-4">
                    <div className="flex items-center justify-center text-sm text-gray-600">
                      <GraduationCap className="w-4 h-4 mr-2" />
                      {member.school}
                    </div>
                    <div className="flex items-center justify-center text-sm text-gray-600">
                      <BookOpen className="w-4 h-4 mr-2" />
                      {member.specialty}
                    </div>
                  </div>

                  <div className="space-y-1">
                    {member.achievements.map((achievement, achievementIndex) => (
                      <div key={achievementIndex} className="text-[11px] md:text-xs text-gray-600 bg-gray-50 px-2 py-1 rounded">
                        {achievement}
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* School Information */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Trường THPT Chuyên Lê Quý Đôn
          </h3>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* School Overview */}
            <Card className="p-8 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-6">
                <GraduationCap className="w-8 h-8 text-blue-600 mr-3" />
                <h4 className="text-2xl font-bold text-gray-900">{schoolInfo.name}</h4>
              </div>

              <div className="space-y-4 mb-6">
                <div className="flex items-center text-gray-600">
                  <MapPin className="w-5 h-5 mr-3 text-gray-400" />
                  <span>{schoolInfo.location}</span>
                </div>
                <div className="flex items-center text-gray-600">
                  <Award className="w-5 h-5 mr-3 text-gray-400" />
                  <span>{schoolInfo.type}</span>
                </div>
              </div>

              <div>
                <h5 className="font-semibold text-gray-900 mb-3">Các chuyên ngành:</h5>
                <div className="flex flex-wrap gap-2">
                  {schoolInfo.specialties.map((specialty, index) => (
                    <Badge key={index} className="bg-blue-50 text-blue-700">
                      {specialty}
                    </Badge>
                  ))}
                </div>
              </div>
            </Card>

            {/* Achievements */}
            
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <Card className="p-8 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200">
            <UsersIcon className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Tham gia cùng đội Cá Vàng
            </h3>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              Chúng tôi luôn tìm kiếm những tài năng trẻ đam mê công nghệ và sức khỏe cộng đồng
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="primary" className="shadow-md" asChild>
                <a href="https://mail.google.com/mail/?view=cm&fs=1&to=ledinhphuc1408@gmail.com" target="_blank" rel="noopener noreferrer">
                  <Mail className="w-4 h-4 mr-2 text-white" />
                  Liên hệ hợp tác
                </a>
              </Button>
              <Button variant="primaryOutline" asChild>
                <a href="https://www.facebook.com/fucdin" target="_blank" rel="noopener noreferrer">
                  <BookOpen className="w-4 h-4 mr-2" />
                  Tìm hiểu thêm
                </a>
              </Button>
            </div>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
