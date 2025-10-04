import Image from "next/image";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";

export const Header = () => {
  return (
    <div className="w-full flex flex-col items-center justify-center bg-green-400 rounded-3xl shadow-lg py-8 mb-10 mt-2 relative">
      {/* Hamburger menu for mobile */}
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
      <div className="flex items-center gap-6 mb-2">
        
        <h1 className="font-extrabold text-3xl md:text-4xl text-white tracking-wide drop-shadow-lg">
          Trò chơi cải thiện trí nhớ
        </h1>
      </div>
      <p className="text-white text-lg font-semibold opacity-80">Chọn một trò chơi để bắt đầu luyện tập trí nhớ của bạn</p>
    </div>
  );
};


