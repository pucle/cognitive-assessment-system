"use client";

import { useAuth } from "../providers/SimpleAuthProvider";

export function useSafeUser() {
  const auth = useAuth();
  
  return {
    user: auth.user,
    isLoaded: auth.isLoaded,
    isClerkAvailable: false // Demo mode
  };
}
