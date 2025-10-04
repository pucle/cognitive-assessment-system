import MMSEv2Assessment from '@/components/mmse-v2-assessment'
import { Menu, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Sidebar } from "@/components/sidebar"
import Link from "next/link"

export default function MMSEv2Page() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header with hamburger menu */}
      <div className="sticky top-0 z-50 bg-blue-100/95 backdrop-blur-sm border-b border-blue-300 p-2">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-2">
            <div className="md:hidden">
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <Menu className="h-5 w-5 text-blue-700" />
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
                <ArrowLeft className="h-5 w-5 text-blue-700" />
              </Button>
            </Link>
          </div>
          <h1 className="font-bold text-base text-blue-800">
            MMSE Assessment v2.0
          </h1>
          <div />
        </div>
      </div>
      <MMSEv2Assessment />
    </div>
  )
}

export const metadata = {
  title: 'MMSE Assessment v2.0 | Cognitive Assessment System',
  description: 'Advanced MMSE cognitive assessment using AI and audio analysis',
}
