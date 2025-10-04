CREATE TABLE "users" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"age" text NOT NULL,
	"gender" text NOT NULL,
	"email" text NOT NULL,
	"phone" text,
	"avatar" text,
	"title" text,
	"imageSrc" text,
	"mmseScore" text,
	"created_at" timestamp with time zone DEFAULT now(),
	"updated_at" timestamp with time zone DEFAULT now(),
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
ALTER TABLE "user_reports" ALTER COLUMN "age" SET DATA TYPE text;