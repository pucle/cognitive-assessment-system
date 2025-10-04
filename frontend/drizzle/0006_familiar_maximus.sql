ALTER TABLE "questions" ADD COLUMN "session_id" text NOT NULL;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "question_id" text NOT NULL;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "question_content" text NOT NULL;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "audio_file" text;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "auto_transcript" text;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "manual_transcript" text;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "linguistic_analysis" jsonb;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "audio_features" jsonb;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "processed_at" timestamp with time zone;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "created_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "user_name" text;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "user_age" integer;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "user_education" integer;--> statement-breakpoint
ALTER TABLE "questions" ADD COLUMN "user_email" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "session_id" text NOT NULL;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "user_id" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "detailed_results" jsonb;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "chart_data" jsonb;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "exercise_recommendations" jsonb;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "created_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "user_name" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "user_age" integer;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "user_education" integer;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "user_email" text;--> statement-breakpoint
ALTER TABLE "stats" ADD COLUMN "audio_files" jsonb;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "session_id" text NOT NULL;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "question_id" text NOT NULL;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "question_content" text NOT NULL;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "audio_file" text;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "auto_transcript" text;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "raw_audio_features" jsonb;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "created_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "expires_at" timestamp with time zone;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "user_name" text;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "user_age" integer;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "user_education" integer;--> statement-breakpoint
ALTER TABLE "temp_questions" ADD COLUMN "user_email" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "name" text NOT NULL;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "age" text NOT NULL;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "gender" text NOT NULL;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "phone" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "avatar" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "title" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "imageSrc" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "mmseScore" text;--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "created_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "users" ADD COLUMN "updated_at" timestamp with time zone DEFAULT now();--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "sessionId";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "questionId";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "questionContent";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "audioFile";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "autoTranscript";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "manualTranscript";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "linguisticAnalysis";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "audioFeatures";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "processedAt";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "createdAt";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "userName";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "userAge";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "userEducation";--> statement-breakpoint
ALTER TABLE "questions" DROP COLUMN "userEmail";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "sessionId";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "userId";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "detailedResults";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "chartData";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "exerciseRecommendations";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "createdAt";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "userName";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "userAge";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "userEducation";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "userEmail";--> statement-breakpoint
ALTER TABLE "stats" DROP COLUMN "audioFiles";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "sessionId";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "questionId";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "questionContent";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "audioFile";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "autoTranscript";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "rawAudioFeatures";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "createdAt";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "expiresAt";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "userName";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "userAge";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "userEducation";--> statement-breakpoint
ALTER TABLE "temp_questions" DROP COLUMN "userEmail";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "profile";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "mode";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "clerkId";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "createdAt";--> statement-breakpoint
ALTER TABLE "users" DROP COLUMN "updatedAt";