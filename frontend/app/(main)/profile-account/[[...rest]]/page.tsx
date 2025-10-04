"use client";

import { UserProfile } from "@clerk/nextjs";
import { Menu, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/sidebar";
import Link from "next/link";

const ProfileClerkPage = () => {
  return (
    <div className="h-full bg-slate-50 p-2 sm:p-3 lg:p-4">
      {/* Header with hamburger menu */}
      <div className="sticky top-0 z-50 bg-slate-100/95 backdrop-blur-sm border-b border-slate-300 p-2">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-2">
            <div className="md:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <Menu className="h-5 w-5 text-slate-700" />
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
            <Link href="/menu">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-5 w-5 text-slate-700" />
              </Button>
            </Link>
          </div>
          <h1 className="font-bold text-base text-slate-800">
            Tài khoản
          </h1>
          <div />
        </div>
      </div>
      <div className="max-w-5xl mx-auto">
        <div className="bg-white shadow-md rounded-xl p-4 sm:p-5 lg:p-6">
          <h2 className="text-lg sm:text-xl font-bold text-slate-800 mb-4 sm:mb-6">Tài khoản</h2>
          <UserProfile
            appearance={{
              elements: {
                rootBox: "w-full",
                card: "shadow-none p-0",
                navbar: "hidden",
                pageScrollBox: "p-0"
              }
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default ProfileClerkPage;
