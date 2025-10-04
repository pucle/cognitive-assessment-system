import * as React from "react"
import { cn } from "@/lib/utils"

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "destructive" | "info" | "success" | "warning"
}

export const Alert = React.forwardRef<HTMLDivElement, AlertProps>(({ className, variant = "default", ...props }, ref) => {
  const variantClass = {
    default: "bg-white border-blue-100 text-gray-800",
    destructive: "bg-red-50 border-red-200 text-red-800",
    info: "bg-blue-50 border-blue-200 text-blue-800",
    success: "bg-green-50 border-green-200 text-green-800",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-800",
  }[variant]

  return (
    <div
      ref={ref}
      role="alert"
      className={cn(
        "w-full rounded-xl border px-4 py-3 text-sm",
        variantClass,
        className
      )}
      {...props}
    />
  )
})
Alert.displayName = "Alert"

export const AlertDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn("mt-1 text-sm leading-relaxed", className)} {...props} />
  )
)
AlertDescription.displayName = "AlertDescription"
