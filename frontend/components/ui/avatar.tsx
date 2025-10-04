import * as React from "react";
import { cn } from "@/lib/utils";

interface AvatarProps {
  src?: string;
  alt?: string;
  size?: number; // default 64
  fallback?: string;
  className?: string;
}

export const Avatar = ({ src, alt, size = 64, fallback, className }: AvatarProps) => {
  return (
    <div
      className={cn(
        "rounded-full shadow-lg border-4 border-blue-300 bg-white flex items-center justify-center overflow-hidden",
        className
      )}
      style={{ width: size, height: size }}
    >
      {src ? (
        <img
          src={src}
          alt={alt || "Avatar"}
          className="w-full h-full object-cover rounded-full"
        />
      ) : (
        <span className="text-2xl font-bold text-blue-500">
          {fallback || "?"}
        </span>
      )}
    </div>
  );
};
