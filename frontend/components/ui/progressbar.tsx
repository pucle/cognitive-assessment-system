import React from "react";
import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number; // actual value
  maxValue?: number; // maximum value (default 30 for MMSE)
  className?: string;
}

export const ProgressBar = ({ value, maxValue = 30, className }: ProgressBarProps) => {
  // Convert value to percentage based on maxValue
  const percentageValue = (value / maxValue) * 100;

  // Thresholds based on MMSE scale (17, 23, 27 out of 30)
  const threshold1 = (17 / maxValue) * 100;
  const threshold2 = (23 / maxValue) * 100;
  const threshold3 = (27 / maxValue) * 100;

  const safeValue = Math.max(0, Math.min(percentageValue, 100));

  let colorClass = "from-rose-400 to-amber-500";
  if (safeValue > threshold1 && safeValue < threshold2) {
    colorClass = "from-amber-500 to-orange-500";
  } else if (safeValue >= threshold2 && safeValue < threshold3) {
    colorClass = "from-orange-500 to-amber-400";
  } else if (safeValue >= threshold3) {
    colorClass = "from-amber-500 to-rose-500";
  }

  return (
    <div className={cn("w-full h-2.5 bg-white/60 rounded-full overflow-hidden", className)}>
      <div
        className={`h-full bg-gradient-to-r ${colorClass} transition-[width] duration-500`}
        style={{ width: `${safeValue}%` }}
      />
    </div>
  );
};
