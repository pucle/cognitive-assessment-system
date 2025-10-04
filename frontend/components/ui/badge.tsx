import * as React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}

export const Badge = ({ children, icon, className }: BadgeProps) => {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 bg-gradient-to-r from-amber-400 to-rose-400 text-white font-semibold rounded-full px-3.5 py-1.5 text-sm shadow",
        className
      )}
    >
      {icon && <span className="text-lg">{icon}</span>}
      {children}
    </span>
  );
};
