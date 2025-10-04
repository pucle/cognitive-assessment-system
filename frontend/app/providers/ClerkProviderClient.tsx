"use client";

import { ClerkProvider } from "@clerk/nextjs";
import { ReactNode } from "react";

interface ClerkProviderClientProps {
	children: ReactNode;
}

export function ClerkProviderClient({ children }: ClerkProviderClientProps) {
	// Check if we have a valid Clerk publishable key
	const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

	// If no valid key is provided, render children without ClerkProvider
	if (!publishableKey || publishableKey.includes('placeholder') || publishableKey === '') {
		console.warn('Clerk: No valid publishable key provided. Authentication disabled.');
		return <>{children}</>;
	}

	return (
		<ClerkProvider
			publishableKey={publishableKey}
			// Use root as a safe default after-auth redirect; the page will open the profile form if needed
			afterSignInUrl="/"
			afterSignUpUrl="/"
		>
			{children}
		</ClerkProvider>
	);
}
