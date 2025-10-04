CREATE TABLE "user_reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"imageSrc" text NOT NULL,
	"name" text NOT NULL,
	"gender" text NOT NULL,
	"age" integer NOT NULL,
	"email" text NOT NULL,
	"phone" text NOT NULL,
	"mmseScore" integer NOT NULL,
	"createdAt" timestamp with time zone DEFAULT now() NOT NULL
);
