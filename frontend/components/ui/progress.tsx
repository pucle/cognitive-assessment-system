import * as React from "react"
import { cn } from "@/lib/utils"

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number // 0..100
}

export const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(({ className, value = 0, ...props }, ref) => {
  const safe = Math.max(0, Math.min(100, value))
  return (
    <div ref={ref} className={cn("w-full h-2 bg-gray-200 rounded-full overflow-hidden", className)} {...props}>
      <div className="h-full bg-gradient-to-r from-amber-500 to-rose-500 transition-[width] duration-300" style={{ width: `${safe}%` }} />
    </div>
  )
})
Progress.displayName = "Progress"
