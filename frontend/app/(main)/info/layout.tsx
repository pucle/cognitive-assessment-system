import { MobileSidebar } from "@/components/mobile-sidebar";

export default function InfoLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="h-full w-full">
      {/* Mobile hamburger menu (hidden on large screens) */}
      <div className="fixed top-4 right-4 z-[100] lg:hidden">
        <MobileSidebar />
      </div>
      {children}
    </div>
  );
}
