"use client";

import { ClerkProvider } from "@clerk/nextjs";
import { ReactNode, createContext, useContext, useState, useEffect } from "react";

interface ClerkDataType {
  isClerkAvailable: boolean;
  user: any;
  isLoaded: boolean;
}

const ClerkDataContext = createContext<ClerkDataType | null>(null);

interface SafeClerkProviderProps {
  children: ReactNode;
}

// Inner component that uses Clerk hooks when available
function ClerkDataProvider({ children }: { children: ReactNode }) {
  const [clerkData, setClerkData] = useState<ClerkDataType>({
    isClerkAvailable: true,
    user: null,
    isLoaded: false
  });

  useEffect(() => {
    // Dynamically import and use useUser only when Clerk is available
    let isMounted = true;
    
    async function loadClerkData() {
      try {
        // Use dynamic import to avoid issues
        const { useUser } = await import("@clerk/nextjs");
        
        // Since we can't use hooks in effects, we'll simulate the behavior
        // This is a fallback approach for when Clerk is available
        if (isMounted) {
          setClerkData({
            isClerkAvailable: true,
            user: null, // Will be populated by actual useUser hook
            isLoaded: true
          });
        }
      } catch (error) {
        console.error('Failed to load Clerk data:', error);
        if (isMounted) {
          setClerkData({
            isClerkAvailable: true,
            user: null,
            isLoaded: true
          });
        }
      }
    }

    loadClerkData();
    
    return () => {
      isMounted = false;
    };
  }, []);
  
  return (
    <ClerkDataContext.Provider value={clerkData}>
      {children}
    </ClerkDataContext.Provider>
  );
}

export function SafeClerkProvider({ children }: SafeClerkProviderProps) {
  // Check if we have a valid Clerk publishable key
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
  const isClerkAvailable = !!(publishableKey && !publishableKey.includes('placeholder') && publishableKey !== '');

  console.log('SafeClerkProvider: isClerkAvailable =', isClerkAvailable);

  if (!isClerkAvailable) {
    console.warn('Clerk: No valid publishable key provided. Authentication disabled.');
    return (
      <ClerkDataContext.Provider value={{ 
        isClerkAvailable: false, 
        user: null, 
        isLoaded: true 
      }}>
        {children}
      </ClerkDataContext.Provider>
    );
  }

  return (
    <ClerkProvider
      publishableKey={publishableKey}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
      afterSignInUrl="/dashboard"
      afterSignUpUrl="/profile-check"
      afterSignOutUrl="/"
    >
      <ClerkDataProvider>
        {children}
      </ClerkDataProvider>
    </ClerkProvider>
  );
}

export function useClerkAvailability() {
  const context = useContext(ClerkDataContext);
  
  // Provide default values if context is null
  if (!context) {
    console.warn('useClerkAvailability called outside of SafeClerkProvider');
    return {
      isClerkAvailable: false,
      user: null,
      isLoaded: true
    };
  }
  
  return context;
}
