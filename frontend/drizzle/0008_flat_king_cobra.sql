ALTER TABLE "training_samples" ALTER COLUMN "sessionId" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "training_samples" ALTER COLUMN "userEmail" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "training_samples" ALTER COLUMN "questionId" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "training_samples" ALTER COLUMN "questionText" DROP NOT NULL;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "userId" text;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "userInfo" jsonb;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "startedAt" timestamp with time zone;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "completedAt" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "totalQuestions" integer DEFAULT 0;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "answeredQuestions" integer DEFAULT 0;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "completionRate" real;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "memoryScore" real;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "cognitiveScore" real;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "finalMmseScore" integer;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "overallGptScore" real;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "questionResults" jsonb;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "audioFiles" jsonb;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "recordingsPath" text;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "cognitiveAnalysis" jsonb;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "audioFeatures" jsonb;--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "status" text DEFAULT 'completed';--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "usageMode" text DEFAULT 'personal';--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "assessmentType" text DEFAULT 'cognitive';--> statement-breakpoint
ALTER TABLE "training_samples" ADD COLUMN "updatedAt" timestamp with time zone DEFAULT now();