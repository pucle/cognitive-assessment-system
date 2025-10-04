"use client";

import { ReactNode, createContext, useContext, useState } from "react";

interface AuthContextType {
  user: any;
  isLoaded: boolean;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  isLoaded: true,
  isAuthenticated: false
});

interface SimpleAuthProviderProps {
  children: ReactNode;
}

export function SimpleAuthProvider({ children }: SimpleAuthProviderProps) {
  // For demo purposes, we'll use a simple state-based auth
  const [authState] = useState<AuthContextType>({
    user: null,
    isLoaded: true,
    isAuthenticated: false
  });

  return (
    <AuthContext.Provider value={authState}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
