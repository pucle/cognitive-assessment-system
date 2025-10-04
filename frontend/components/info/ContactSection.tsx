"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Mail,
  Phone,
  MapPin,
  Send,
  CheckCircle,
  Loader2,
  MessageSquare,
  Users,
  Stethoscope,
  FileText
} from "lucide-react";
import { useState } from "react";

export default function ContactSection() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    category: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const contactCategories = [
    { value: 'research', label: 'Hợp tác nghiên cứu', icon: FileText },
    { value: 'clinical', label: 'Tham gia thử nghiệm', icon: Stethoscope },
    { value: 'collaboration', label: 'Hợp tác khác', icon: Users },
    { value: 'general', label: 'Câu hỏi khác', icon: MessageSquare }
  ];

  const directContacts = [
    {
      type: 'Email',
      value: 'cavang.project@gmail.com',
      description: 'Liên hệ chính thức',
      icon: Mail
    },
    {
      type: 'Điện thoại',
      value: '+84 xxx xxx xxx',
      description: 'Hotline hỗ trợ',
      icon: Phone
    },
    {
      type: 'Địa chỉ',
      value: 'THPT Chuyên Lê Quý Đôn, Tp. Hồ Chí Minh',
      description: 'Trụ sở chính',
      icon: MapPin
    }
  ];

  const collaborationOpportunities = [
    {
      title: 'Hợp tác nghiên cứu',
      description: 'Chia sẻ dữ liệu, phương pháp nghiên cứu, hoặc phát triển mô hình chung',
      icon: FileText,
      benefits: ['Truy cập dữ liệu', 'Công bố chung', 'Hỗ trợ kỹ thuật']
    },
    {
      title: 'Tham gia thử nghiệm',
      description: 'Cung cấp dữ liệu bệnh nhân hoặc tham gia validation lâm sàng',
      icon: Stethoscope,
      benefits: ['Đóng góp khoa học', 'Truy cập sớm', 'Hỗ trợ chuyên môn']
    },
    {
      title: 'Hợp tác công nghệ',
      description: 'Tích hợp công nghệ AI vào hệ thống bệnh viện hoặc phòng khám',
      icon: Users,
      benefits: ['Giải pháp sẵn sàng', 'Hỗ trợ triển khai', 'Đào tạo đội ngũ']
    }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (!res.ok || !data?.success) throw new Error(data?.error || 'Failed');
    } catch (err) {
      console.error('Contact submit failed', err);
    }

    setIsSubmitting(false);
    setIsSubmitted(true);

    // Reset form after 3 seconds
    setTimeout(() => {
      setIsSubmitted(false);
      setFormData({
        name: '',
        email: '',
        subject: '',
        category: '',
        message: ''
      });
    }, 3000);
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
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
          <Badge  className="mb-4 bg-green-100 text-green-800">
            Liên hệ
          </Badge>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Kết nối cùng chúng tôi
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Hãy liên hệ để cùng nhau phát triển giải pháp công nghệ cho chăm sóc sức khỏe cộng đồng
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <Card className="p-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Gửi tin nhắn</h3>

              {isSubmitted ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-8"
                >
                  <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
                  <h4 className="text-xl font-semibold text-gray-900 mb-2">
                    Cảm ơn bạn đã liên hệ!
                  </h4>
                  <p className="text-gray-600">
                    Chúng tôi sẽ phản hồi trong vòng 24-48 giờ.
                  </p>
                </motion.div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="name">Họ và tên *</Label>
                      <Input
                        id="name"
                        type="text"
                        required
                        value={formData.name}
                        onChange={(e) => handleInputChange('name', e.target.value)}
                        placeholder="Nhập họ và tên"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="email">Email *</Label>
                      <Input
                        id="email"
                        type="email"
                        required
                        value={formData.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        placeholder="email@example.com"
                        className="mt-1"
                      />
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="category">Loại liên hệ *</Label>
                    <Select
                      value={formData.category}
                      onValueChange={(value) => handleInputChange('category', value)}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Chọn loại liên hệ" />
                      </SelectTrigger>
                      <SelectContent>
                        {contactCategories.map((category) => (
                          <SelectItem key={category.value} value={category.value}>
                            <div className="flex items-center">
                              <category.icon className="w-4 h-4 mr-2" />
                              {category.label}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="subject">Tiêu đề</Label>
                    <Input
                      id="subject"
                      type="text"
                      value={formData.subject}
                      onChange={(e) => handleInputChange('subject', e.target.value)}
                      placeholder="Tiêu đề tin nhắn"
                      className="mt-1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="message">Nội dung tin nhắn *</Label>
                    <textarea
                      id="message"
                      required
                      value={formData.message}
                      onChange={(e) => handleInputChange('message', e.target.value)}
                      placeholder="Mô tả chi tiết về yêu cầu của bạn..."
                      className="mt-1 w-full h-32 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    />
                  </div>

                  <Button
                    type="submit"
                    disabled={isSubmitting}
                    variant="primary"
                    className="w-full text-white"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Đang gửi...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" />
                        Gửi tin nhắn
                      </>
                    )}
                  </Button>
                </form>
              )}
            </Card>
          </motion.div>

          
        </div>
      </div>
    </section>
  );
}
