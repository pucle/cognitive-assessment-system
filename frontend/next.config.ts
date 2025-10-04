import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	outputFileTracingRoot: __dirname,
	
	// Production optimizations
	compress: true,
	poweredByHeader: false,
	generateEtags: false,
	
	// Image optimization
	images: {
		domains: [
			'localhost',
			'*.vercel-storage.com',
			'*.public.blob.vercel-storage.com',
			'*.herokuapp.com'
		],
		remotePatterns: [
			{
				protocol: 'https',
				hostname: '*.public.blob.vercel-storage.com',
			},
			{
				protocol: 'https',
				hostname: '*.herokuapp.com',
			},
		],
		formats: ['image/avif', 'image/webp'],
		deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
		imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
	},
	
	// Experimental features
	experimental: {
		optimizeCss: true,
	},

	// Server external packages
	serverExternalPackages: ['@neondatabase/serverless'],
	
	// Environment variables
	env: {
		DATABASE_URL: process.env.DATABASE_URL,
		NEON_DATABASE_URL: process.env.NEON_DATABASE_URL,
		NEXT_PUBLIC_PYTHON_BACKEND_URL: process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL,
	},
	
	// Headers for security and performance
	async headers() {
		return [
			{
				source: '/(.*)',
				headers: [
					{
						key: 'X-Frame-Options',
						value: 'DENY',
					},
					{
						key: 'X-Content-Type-Options',
						value: 'nosniff',
					},
					{
						key: 'Referrer-Policy',
						value: 'origin-when-cross-origin',
					},
				],
			},
			{
				source: '/api/(.*)',
				headers: [
					{
						key: 'Access-Control-Allow-Origin',
						value: process.env.NODE_ENV === 'production' 
							? process.env.NEXT_PUBLIC_APP_URL || 'https://*.herokuapp.com'
							: '*',
					},
					{
						key: 'Access-Control-Allow-Methods',
						value: 'GET, POST, PUT, DELETE, OPTIONS',
					},
					{
						key: 'Access-Control-Allow-Headers',
						value: 'Content-Type, Authorization',
					},
				],
			},
		];
	},
	
	// Webpack configuration for production
	webpack: (config, { isServer, dev }) => {
		// Production optimizations
		if (!dev) {
			config.optimization.minimize = true;
		}
		
		// Handle audio files
		config.module.rules.push({
			test: /\.(mp3|wav|ogg|flac)$/,
			use: {
				loader: 'file-loader',
				options: {
					publicPath: '/_next/static/audio/',
					outputPath: `${isServer ? '../' : ''}static/audio/`,
				},
			},
		});
		
		return config;
	},
};

export default nextConfig;
