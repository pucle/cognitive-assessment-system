ALTER TABLE "sessions" ADD COLUMN "user_id" text;--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "start_time" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "end_time" timestamp with time zone;--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "total_score" real;--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "mmse_score" integer;--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "cognitive_level" "cognitive_level";--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "email_sent" integer DEFAULT 0;--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "created_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "sessions" ADD COLUMN "updated_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "userId";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "startTime";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "endTime";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "totalScore";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "mmseScore";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "cognitiveLevel";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "emailSent";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "createdAt";--> statement-breakpoint
ALTER TABLE "sessions" DROP COLUMN "updatedAt";