//app/main/game/page.tsx
import { Header } from "./header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Image from "next/image";
import Link from "next/link";

const games = [
  {
    title: "Gợi từ theo chủ đề",
    description: "Luyện trí nhớ và liên kết từ vựng theo các chủ đề quen thuộc.",
    icon: "/file.svg",
    href: "/game/word-cue",
    color: "from-[#F4A261] via-[#E88D4D] to-[#E67635]",
    bgColor: "bg-gradient-to-br from-[#FAE6D0] via-[#F5D7BE] to-[#F0C8A8]",
    badge: "🐠",
  },
  {
    title: "Kể chuyện",
    description: "Phát triển trí nhớ dài hạn và khả năng sáng tạo qua kể chuyện.",
    icon: "/hero.svg",
    href: "/game/story-telling",
    color: "from-[#F4A261] via-[#E88D4D] to-[#E67635]",
    bgColor: "bg-gradient-to-br from-[#FAE6D0] via-[#F5D7BE] to-[#F0C8A8]",
    badge: "📖",
  },
  {
    title: "Mô tả tranh, ảnh",
    description: "Rèn trí nhớ hình ảnh và khả năng quan sát chi tiết.",
    icon: "/window.svg",
    href: "/game/picture-description",
    color: "from-[#F4A261] via-[#E88D4D] to-[#E67635]",
    bgColor: "bg-gradient-to-br from-[#FAE6D0] via-[#F5D7BE] to-[#F0C8A8]",
    badge: "🖼️",
  },
  {
    title: "Tán gẫu",
    description: "Tăng trí nhớ giao tiếp và phản xạ ngôn ngữ qua trò chuyện.",
    icon: "/chat.svg",
    href: "/game/chatting",
    color: "from-[#F4A261] via-[#E88D4D] to-[#E67635]",
    bgColor: "bg-gradient-to-br from-[#FAE6D0] via-[#F5D7BE] to-[#F0C8A8]",
    badge: "💬",
  },
];

export default function GamePage() {
  return (
    <div className="min-h-screen max-h-[150vh] w-full overflow-auto" style={{
      background: 'linear-gradient(135deg, #FEF3E2 0%, #FAE6D0 50%, #F5D7BE 100%)'
    }}>
      <Header />
      <div className="w-full max-w-6xl mx-auto flex flex-col gap-4 sm:gap-6 lg:gap-8 px-2 sm:px-3 lg:px-4 py-3 sm:py-4 lg:py-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2 sm:gap-3 lg:gap-4 items-stretch">
          {games.map((game, idx) => (
            <Link key={idx} href={game.href} className="group h-full">
              <Card
                className={`flex flex-col justify-between items-center ${game.bgColor} shadow-xl
                  p-4 sm:p-5 lg:p-6 rounded-2xl h-full transition-all duration-300 hover:shadow-2xl hover:-translate-y-1
                  relative overflow-hidden`} style={{
                  border: '2px solid #F4A261',
                  boxShadow: '0 8px 16px rgba(244, 162, 97, 0.2)',
                  color: '#8B6D57'
                }}
              >
                {/* Trang trí góc */}
                <div
                  className={`absolute top-0 right-0 w-12 h-12 sm:w-16 sm:h-16 lg:w-20 lg:h-20 bg-gradient-to-br ${game.color} opacity-20 rounded-bl-full`}
                ></div>

                {/* Nội dung */}
                <div className="flex flex-col items-center flex-1 justify-center text-center px-2">
                  {/* Icon */}
                  <div
                    className={`bg-gradient-to-br ${game.color} rounded-xl flex items-center justify-center
                      w-16 h-16 sm:w-20 sm:h-20 lg:w-24 lg:h-24 shadow-md group-hover:shadow-lg transition-all duration-300
                      group-hover:scale-105 border-2 sm:border-3 lg:border-4 border-white overflow-hidden mb-3 sm:mb-4 lg:mb-5`}
                  >
                    <Image
                      src={game.icon}
                      alt={game.title}
                      width={64}
                      height={64}
                      className="w-12 h-12 sm:w-16 sm:h-16 lg:w-20 lg:h-20 object-contain"
                    />
                  </div>

                  {/* Badge */}
                  <Badge
                    className={`mb-2 sm:mb-3 px-2 sm:px-3 lg:px-4 py-1 text-base sm:text-lg lg:text-xl bg-gradient-to-r ${game.color} shadow rounded-full`}
                  >
                    {game.badge}
                  </Badge>

                  {/* Tiêu đề */}
                  <h3
                    className={`font-extrabold text-lg sm:text-xl lg:text-2xl mb-2 sm:mb-3 leading-snug transition-all group-hover:scale-[1.02]`}
                    style={{ color: '#B8763E' }}
                  >
                    {game.title}
                  </h3>

                  {/* Mô tả */}
                  <p
                    className="text-sm sm:text-base leading-relaxed transition-colors"
                    style={{ color: '#8B6D57' }}
                  >
                    {game.description}
                  </p>
                </div>

                {/* Hiệu ứng click sáng */}
                <span className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
