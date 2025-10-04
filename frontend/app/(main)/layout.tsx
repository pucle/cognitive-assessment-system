import type { PropsWithChildren } from "react";

import { Sidebar } from "@/components/sidebar";

const MainLayout = ({ children }: PropsWithChildren) => {
  return (
    <>
      <Sidebar className="hidden lg:flex" />
      <main className="h-full lg:pl-[256px]">
        <div className="h-full w-full pt-6">{children}</div>
      </main>
    </>
  );
};

export default MainLayout;