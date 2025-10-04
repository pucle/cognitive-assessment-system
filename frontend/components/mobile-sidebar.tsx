"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";
import { Menu } from "lucide-react";

export const MobileSidebar = () => {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <button
          aria-label="Mở menu điều hướng"
          className="inline-flex items-center justify-center rounded-xl p-2 bg-blue-600 text-white shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <Menu className="w-5 h-5" />
        </button>
      </SheetTrigger>
      <SheetContent
        className="p-0 z-[100] h-full max-w-none"
        side="right"
      >
        <SheetHeader>
          <SheetTitle className="sr-only">Navigation Sidebar</SheetTitle>
        </SheetHeader>
        
        <Sidebar className="h-full w-full lg:static lg:left-auto lg:top-auto lg:w-full" />
      </SheetContent>
    </Sheet>
  );
};
