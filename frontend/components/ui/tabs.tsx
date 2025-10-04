"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface TabsContextValue {
  value: string
  setValue: (v: string) => void
}

const TabsContext = React.createContext<TabsContextValue | null>(null)

export function Tabs({ value, onValueChange, className, children }: { value: string; onValueChange?: (v: string) => void; className?: string; children: React.ReactNode }) {
  const [internal, setInternal] = React.useState(value)

  React.useEffect(() => setInternal(value), [value])

  const setValue = (v: string) => {
    setInternal(v)
    onValueChange?.(v)
  }

  return (
    <TabsContext.Provider value={{ value: internal, setValue }}>
      <div className={cn("w-full", className)}>{children}</div>
    </TabsContext.Provider>
  )
}

export function TabsList({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("inline-flex items-center gap-2 bg-white/70 backdrop-blur-sm border border-white/70 rounded-full p-1", className)}>{children}</div>
}

export function TabsTrigger({ value, className, children }: { value: string; className?: string; children: React.ReactNode }) {
  const ctx = React.useContext(TabsContext)
  if (!ctx) return null
  const active = ctx.value === value
  return (
    <button
      type="button"
      onClick={() => ctx.setValue(value)}
      className={cn(
        "px-3.5 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition-all",
        active ? "bg-gradient-to-r from-amber-500 to-rose-500 text-white shadow" : "text-gray-700 hover:bg-white"
      )}
      aria-selected={active}
      role="tab"
    >
      {children}
    </button>
  )
}

export function TabsContent({ value, className, children }: { value: string; className?: string; children: React.ReactNode }) {
  const ctx = React.useContext(TabsContext)
  if (!ctx) return null
  if (ctx.value !== value) return null
  return <div className={cn("mt-4", className)} role="tabpanel">{children}</div>
}
