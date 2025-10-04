"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

type Tab = {
  key: string;
  label: string;
};

interface PillTabsProps {
  tabs: Tab[];
  value: string;
  onChange: (key: string) => void;
  className?: string;
}

export function PillTabs({ tabs, value, onChange, className }: PillTabsProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 bg-white/70 backdrop-blur-sm border border-white/70 rounded-full p-1",
        className
      )}
      role="tablist"
    >
      {tabs.map((t) => {
        const active = t.key === value;
        return (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(t.key)}
            className={cn(
              "px-3.5 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition-all",
              active
                ? "bg-gradient-to-r from-amber-500 to-rose-500 text-white shadow"
                : "text-gray-700 hover:bg-white"
            )}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}


