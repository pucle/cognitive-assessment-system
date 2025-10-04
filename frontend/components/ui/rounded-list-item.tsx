"use client";

import * as React from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface RoundedListItemProps extends React.HTMLAttributes<HTMLButtonElement> {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
  end?: React.ReactNode;
}

export function RoundedListItem({ icon, title, subtitle, end, className, ...props }: RoundedListItemProps) {
  return (
    <button
      className={cn(
        "w-full flex items-center gap-3 px-3 py-3 sm:px-4 sm:py-3.5 bg-white/80 hover:bg-white rounded-2xl border border-white/60 shadow-sm transition",
        className
      )}
      {...props}
    >
      {icon && <div className="shrink-0 w-10 h-10 rounded-xl bg-gray-100 flex items-center justify-center">{icon}</div>}
      <div className="flex-1 text-left">
        <div className="text-sm font-semibold text-gray-800">{title}</div>
        {subtitle && <div className="text-xs text-gray-500">{subtitle}</div>}
      </div>
      <div className="flex items-center gap-2">
        {end}
        <ChevronRight className="w-4 h-4 text-gray-400" />
      </div>
    </button>
  );
}


