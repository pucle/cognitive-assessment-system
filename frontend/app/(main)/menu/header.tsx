import Image from "next/image";
import { Fish, Waves, Shell, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";

export const Header = () => {
  return (
    <div className="w-full flex flex-col items-center justify-center bg-gradient-to-r from-amber-400 via-orange-300 to-rose-400 rounded-3xl shadow-xl py-6 sm:py-7 mb-6 sm:mb-8 mt-2 relative overflow-hidden">
      {/* Hamburger menu */}
      <div className="md:hidden absolute top-3 left-3 z-20">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="sm" className="text-white hover:bg-white/20">
              <Menu className="h-6 w-6" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-80">
            <SheetHeader>
              <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
            </SheetHeader>
            <Sidebar />
          </SheetContent>
        </Sheet>
      </div>

      {/* Underwater decorations */}
      <div className="absolute top-4 right-8 text-white/20">
        <Fish className="w-8 h-8 rotate-12 animate-bounce" />
      </div>
      <div className="absolute bottom-4 left-8 text-white/20">
        <Shell className="w-6 h-6 animate-pulse" />
      </div>
      <div className="absolute top-1/2 right-16 text-white/15">
        <Waves className="w-5 h-5 animate-pulse" />
      </div>

      <div className="flex items-center gap-6 mb-2 relative z-10">
        <div className="relative">
          <Image src="/mascot.svg" alt="Mascot" width={56} height={56} className="rounded-2xl shadow-md" />
          {/* Glowing effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-2xl opacity-30 animate-pulse"></div>
        </div>
        <h1 className="font-extrabold text-2xl md:text-4xl text-white tracking-wide drop-shadow-lg">
           Menu
        </h1>
      </div>
      <p className="text-white text-base sm:text-lg font-semibold opacity-90 drop-shadow-md text-center max-w-2xl leading-relaxed">
        Khám phá các tính năng tuyệt vời để bắt đầu hành trình cải thiện trí nhớ của bạn dưới đáy biển kỳ diệu
      </p>

      {/* Animated bubbles */}
      <div className="absolute bottom-2 left-1/4 w-2 h-2 bg-white/40 rounded-full animate-bounce"></div>
      <div className="absolute bottom-6 right-1/3 w-1 h-1 bg-white/30 rounded-full animate-bounce" style={{animationDelay: '0.5s'}}></div>
      <div className="absolute bottom-4 right-1/4 w-3 h-3 bg-white/50 rounded-full animate-bounce" style={{animationDelay: '1s'}}></div>
    </div>
  );
};
