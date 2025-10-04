"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface DockAction {
  key: string;
  icon: React.ReactNode;
  onClick?: () => void;
  tooltip?: string;
}

interface BottomDockProps {
  actions: DockAction[];
  className?: string;
}

export function BottomDock({ actions, className }: BottomDockProps) {
  return (
    <div className={cn("fixed inset-x-0 bottom-3 sm:bottom-4 flex justify-center pointer-events-none px-3 sm:px-4", className)}>
      <div className="pointer-events-auto inline-flex items-center gap-3 sm:gap-4 bg-white/80 backdrop-blur-xl rounded-2xl px-3 sm:px-4 py-1.5 sm:py-2 shadow-2xl border border-white/60">
        {actions.map((a) => (
          <button
            key={a.key}
            title={a.tooltip}
            onClick={a.onClick}
            className="w-9 h-9 sm:w-11 sm:h-11 rounded-full bg-gradient-to-tr from-amber-500 to-rose-500 text-white flex items-center justify-center shadow-md hover:shadow-lg active:scale-95"
          >
            {a.icon}
          </button>
        ))}
      </div>
    </div>
  );
}


