import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const isPublicRoute = createRouteMatcher([
  "/",
  "/sign-in(.*)",
  "/sign-up(.*)",
  "/stats(.*)",
  "/cognitive-assessment(.*)",
  "/results(.*)",
  "/menu(.*)"
]);

const isApiRoute = createRouteMatcher([
  "/api/(.*)"
]);

export default clerkMiddleware(async (auth, request) => {
  // Skip authentication for API routes during development
  if (isApiRoute(request)) {
    return;
  }

  // Check if Clerk is properly configured
  try {
    const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
    if (!publishableKey || publishableKey.includes('placeholder') || publishableKey === '') {
      console.warn('Clerk middleware: No valid key, skipping protection');
      return;
    }

    // Protect non-public routes only if Clerk is available
    if (!isPublicRoute(request)) {
      await auth.protect();
    }
  } catch (error) {
    console.warn('Clerk middleware error:', error);
    // Skip protection if Clerk fails
    return;
  }
});

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
};