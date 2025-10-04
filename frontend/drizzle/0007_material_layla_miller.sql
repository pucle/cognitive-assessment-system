CREATE TABLE "contact_messages" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"email" text NOT NULL,
	"subject" text,
	"category" text,
	"message" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now()
);
