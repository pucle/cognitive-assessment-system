-- Tạo bảng users nếu chưa tồn tại
CREATE TABLE IF NOT EXISTS "users" (
    "id" serial PRIMARY KEY NOT NULL,
    "name" text NOT NULL,
    "age" text NOT NULL,
    "gender" text NOT NULL,
    "email" text NOT NULL UNIQUE,
    "phone" text,
    "avatar" text,
    "title" text,
    "imageSrc" text,
    "mmseScore" text,
    "created_at" timestamp with time zone DEFAULT now(),
    "updated_at" timestamp with time zone DEFAULT now()
);
