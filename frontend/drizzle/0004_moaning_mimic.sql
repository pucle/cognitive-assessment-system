ALTER TABLE "questions" ADD COLUMN "userName" text;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "userAge" integer;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "userEducation" integer;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "userEmail" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "userName" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "userAge" integer;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "userEducation" integer;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "userEmail" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "audioFiles" jsonb;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "userName" text;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "userAge" integer;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "userEducation" integer;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "userEmail" text;