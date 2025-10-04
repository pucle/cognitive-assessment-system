CREATE TYPE "public"."cognitive_level" AS ENUM('mild', 'moderate', 'severe', 'normal');--> statement-breakpoint
CREATE TYPE "public"."session_status" AS ENUM('in_progress', 'completed', 'error');--> statement-breakpoint
CREATE TYPE "public"."user_mode" AS ENUM('personal', 'community');--> statement-breakpoint
CREATE TABLE "cognitive_assessment_results" (
	"id" serial PRIMARY KEY NOT NULL,
	"sessionId" text NOT NULL,
	"userId" text,
	"userInfo" jsonb,
	"startedAt" timestamp with time zone,
	"completedAt" timestamp with time zone DEFAULT now(),
	"totalQuestions" integer DEFAULT 0,
	"answeredQuestions" integer DEFAULT 0,
	"completionRate" real,
	"memoryScore" real,
	"cognitiveScore" real,
	"finalMmseScore" integer,
	"overallGptScore" real,
	"questionResults" jsonb,
	"audioFiles" jsonb,
	"recordingsPath" text,
	"cognitiveAnalysis" jsonb,
	"audioFeatures" jsonb,
	"status" text DEFAULT 'completed',
	"usageMode" text DEFAULT 'personal',
	"assessmentType" text DEFAULT 'cognitive',
	"createdAt" timestamp with time zone DEFAULT now(),
	"updatedAt" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "community_assessments" (
	"id" serial PRIMARY KEY NOT NULL,
	"sessionId" text NOT NULL,
	"name" text,
	"email" text NOT NULL,
	"age" text,
	"gender" text,
	"phone" text,
	"status" text DEFAULT 'pending',
	"finalMmse" integer,
	"overallGptScore" integer,
	"resultsJson" text,
	"createdAt" timestamp with time zone DEFAULT now(),
	"updatedAt" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "questions" (
	"id" serial PRIMARY KEY NOT NULL,
	"sessionId" text NOT NULL,
	"questionId" text NOT NULL,
	"questionContent" text NOT NULL,
	"audioFile" text,
	"autoTranscript" text,
	"manualTranscript" text,
	"linguisticAnalysis" jsonb,
	"audioFeatures" jsonb,
	"evaluation" text,
	"feedback" text,
	"score" real,
	"processedAt" timestamp with time zone,
	"createdAt" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"id" serial PRIMARY KEY NOT NULL,
	"userId" text,
	"mode" "user_mode" NOT NULL,
	"status" "session_status" DEFAULT 'in_progress',
	"startTime" timestamp with time zone DEFAULT now(),
	"endTime" timestamp with time zone,
	"totalScore" real,
	"mmseScore" integer,
	"cognitiveLevel" "cognitive_level",
	"emailSent" integer DEFAULT 0,
	"createdAt" timestamp with time zone DEFAULT now(),
	"updatedAt" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "stats" (
	"id" serial PRIMARY KEY NOT NULL,
	"sessionId" text NOT NULL,
	"userId" text,
	"timestamp" timestamp with time zone DEFAULT now(),
	"mode" "user_mode" NOT NULL,
	"summary" jsonb,
	"detailedResults" jsonb,
	"chartData" jsonb,
	"exerciseRecommendations" jsonb,
	"createdAt" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "temp_questions" (
	"id" serial PRIMARY KEY NOT NULL,
	"sessionId" text NOT NULL,
	"questionId" text NOT NULL,
	"questionContent" text NOT NULL,
	"audioFile" text,
	"autoTranscript" text,
	"rawAudioFeatures" jsonb,
	"status" text DEFAULT 'pending',
	"createdAt" timestamp with time zone DEFAULT now(),
	"expiresAt" timestamp with time zone
);
--> statement-breakpoint
CREATE TABLE "training_samples" (
	"id" serial PRIMARY KEY NOT NULL,
	"sessionId" text,
	"userEmail" text NOT NULL,
	"userName" text,
	"questionId" text NOT NULL,
	"questionText" text NOT NULL,
	"audioFilename" text,
	"audioUrl" text,
	"autoTranscript" text,
	"manualTranscript" text,
	"createdAt" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "displayName" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "profile" jsonb;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "mode" "user_mode" DEFAULT 'personal';--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "clerkId" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "createdAt" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "updatedAt" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "name";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "age";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "gender";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "phone";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "avatar";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "title";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "imageSrc";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "mmseScore";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "created_at";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "updated_at";