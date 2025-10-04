"use client";

import { ReactNode, useEffect } from "react";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { ClerkProviderClient } from "./ClerkProviderClient";
import { ErrorBoundary } from "@/components/ErrorBoundary";

interface ClientProvidersProps {
	children: ReactNode;
}

export function ClientProviders({ children }: ClientProvidersProps) {
	return (
		<ErrorBoundary>
			<ClerkProviderClient>
				<LanguageProvider>
					<BilingualWrapper>
						{children}
					</BilingualWrapper>
				</LanguageProvider>
			</ClerkProviderClient>
		</ErrorBoundary>
	);
}

// Simplified wrapper without hooks
function BilingualWrapper({ children }: { children: ReactNode }) {
	useEffect(() => {
		// Update document language attribute and title
		if (typeof window !== 'undefined') {
			const html = document.documentElement;
			const currentLang = html.getAttribute('lang') || 'vi';
			html.setAttribute('lang', currentLang);

			// Update title
			const title = document.title;
			if (!title.includes('Cá Vàng')) {
				document.title = 'Cá Vàng';
			}
		}
	}, []);

	return <>{children}</>;
}


