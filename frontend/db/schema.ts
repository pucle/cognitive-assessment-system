//db/schema.ts - Pipeline System Database Schema
import { serial, pgTable, text, timestamp, integer, jsonb, real, pgEnum, varchar } from "drizzle-orm/pg-core";

// Enums
export const userModeEnum = pgEnum('user_mode', ['personal', 'community']);
export const sessionStatusEnum = pgEnum('session_status', ['in_progress', 'completed', 'error']);
export const cognitiveLevelEnum = pgEnum('cognitive_level', ['mild', 'moderate', 'severe', 'normal']);

// Schema cho bảng users chính (match với database hiện tại)
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  // Legacy columns (giữ lại để tương thích)
  name: text("name").notNull(),
  age: text("age").notNull(),
  gender: text("gender").notNull(),
  email: text("email").notNull().unique(),
  phone: text("phone"),
  avatar: text("avatar"),
  title: text("title"),
  imageSrc: text("imageSrc"),
  mmseScore: text("mmseScore"),
  // New columns
  displayName: text("displayName"),
  // profile: jsonb("profile"), // COMMENTED OUT - not needed, using individual columns instead
  // mode: userModeEnum("mode").default('personal'), // COMMENTED OUT - not needed for current implementation
  // clerkId: text("clerkId"), // COMMENTED OUT - not needed for current implementation
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

// Schema cho bảng sessions (phiên đánh giá)
export const sessions = pgTable("sessions", {
  id: serial("id").primaryKey(),
  userId: text("user_id"), // ánh xạ từ cột user_id trong database
  mode: userModeEnum("mode").notNull(), // personal hoặc community
  status: sessionStatusEnum("status").default('in_progress'),
  startTime: timestamp("start_time", { withTimezone: true }).defaultNow(),
  endTime: timestamp("end_time", { withTimezone: true }),
  totalScore: real("total_score"),
  mmseScore: integer("mmse_score"),
  cognitiveLevel: cognitiveLevelEnum("cognitive_level"),
  emailSent: integer("email_sent").default(0), // 0 = chưa gửi, 1 = đã gửi
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

// Schema cho bảng questions (câu hỏi và kết quả) - match với database
export const questions = pgTable("questions", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  questionId: text("question_id").notNull(), // ID của câu hỏi từ MMSE
  questionContent: text("question_content").notNull(),
  audioFile: text("audio_file"), // path/URL của file audio
  autoTranscript: text("auto_transcript"), // transcript từ Gemini ASR (autotranscribe)
  manualTranscript: text("manual_transcript"), // transcript manual (nếu có)
  linguisticAnalysis: jsonb("linguistic_analysis"), // kết quả phân tích ngôn ngữ từ GPT-4o
  audioFeatures: jsonb("audio_features"), // đặc trưng âm thanh từ Librosa
  evaluation: text("evaluation"), // đánh giá AI
  feedback: text("feedback"), // phản hồi cho user
  score: real("score"), // điểm số của câu hỏi
  processedAt: timestamp("processed_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  // Additional user info fields
  userName: text("user_name"), // tên người dùng
  userAge: integer("user_age"), // tuổi người dùng
  userEducation: integer("user_education"), // số năm học
  userEmail: text("user_email"), // email người dùng
});

// Schema cho bảng stats (thống kê và báo cáo) - match với database
export const stats = pgTable("stats", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  userId: text("user_id"),
  timestamp: timestamp("timestamp", { withTimezone: true }).defaultNow(),
  mode: userModeEnum("mode").notNull(),
  summary: jsonb("summary"), // tổng kết (totalScore, mmseScore, cognitiveLevel, etc.)
  detailedResults: jsonb("detailed_results"), // kết quả chi tiết đầy đủ
  chartData: jsonb("chart_data"), // dữ liệu biểu đồ (chỉ cho personal mode)
  exerciseRecommendations: jsonb("exercise_recommendations"), // gợi ý bài tập (chỉ cho personal mode)
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  // Additional user info fields
  userName: text("user_name"), // tên người dùng
  userAge: integer("user_age"), // tuổi người dùng
  userEducation: integer("user_education"), // số năm học
  userEmail: text("user_email"), // email người dùng
  audioFiles: jsonb("audio_files"), // file ghi âm
});

// Schema cho bảng temp_questions (dữ liệu tạm thời trong quá trình assessment) - match với database
export const tempQuestions = pgTable("temp_questions", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  questionId: text("question_id").notNull(),
  questionContent: text("question_content").notNull(),
  audioFile: text("audio_file"), // file ghi âm
  autoTranscript: text("auto_transcript"), // autotranscribe
  rawAudioFeatures: jsonb("raw_audio_features"), // dữ liệu thô từ Gemini
  status: text("status").default('pending'), // pending, processing, completed
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  expiresAt: timestamp("expires_at", { withTimezone: true }), // auto cleanup
  // Additional user info fields
  userName: text("user_name"), // tên người dùng
  userAge: integer("user_age"), // tuổi người dùng
  userEducation: integer("user_education"), // số năm học
  userEmail: text("user_email"), // email người dùng
});

// Legacy tables for backward compatibility - match với database
export const userReports = pgTable("user_reports", {
  id: serial("id").primaryKey(),
  title: text("title"),
  imageSrc: text("imageSrc"),
  name: text("name"),
  gender: text("gender"),
  age: text("age"),
  email: text("email"),
  phone: text("phone"),
  mmseScore: integer("mmseScore"),
  createdAt: timestamp("createdAt", { withTimezone: true }).defaultNow(),
});

export const trainingSamples = pgTable("training_samples", {
  id: serial("id").primaryKey(),
  // Use snake_case column names to match actual database
  sessionId: varchar("session_id", { length: 255 }).notNull(),
  userId: varchar("user_id", { length: 255 }).notNull(),
  userEmail: varchar("user_email", { length: 255 }),
  userName: varchar("user_name", { length: 255 }),
  questionId: integer("question_id"),
  questionText: text("question_text"),
  audioFilename: varchar("audio_filename", { length: 255 }),
  audioUrl: text("audio_url"),
  autoTranscript: text("auto_transcript"),
  manualTranscript: text("manual_transcript"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

export const communityAssessments = pgTable("community_assessments", {
  id: serial("id").primaryKey(),
  sessionId: text("sessionId").notNull(),
  name: text("name"),
  email: text("email").notNull(),
  age: text("age"),
  gender: text("gender"),
  phone: text("phone"),
  status: text("status").default('pending'),
  finalMmse: integer("finalMmse"),
  overallGptScore: integer("overallGptScore"),
  resultsJson: text("resultsJson"),
  createdAt: timestamp("createdAt", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updatedAt", { withTimezone: true }).defaultNow(),
});

export const cognitiveAssessmentResults = pgTable("cognitive_assessment_results", {
  id: serial("id").primaryKey(),
  sessionId: text("sessionId").notNull(),
  userId: text("userId"),
  userInfo: jsonb("userInfo"),
  startedAt: timestamp("startedAt", { withTimezone: true }),
  completedAt: timestamp("completedAt", { withTimezone: true }).defaultNow(),
  totalQuestions: integer("totalQuestions").default(0),
  answeredQuestions: integer("answeredQuestions").default(0),
  completionRate: real("completionRate"),
  memoryScore: real("memoryScore"),
  cognitiveScore: real("cognitiveScore"),
  finalMmseScore: integer("finalMmseScore"),
  overallGptScore: real("overallGptScore"),
  questionResults: jsonb("questionResults"),
  audioFiles: jsonb("audioFiles"),
  recordingsPath: text("recordingsPath"),
  cognitiveAnalysis: jsonb("cognitiveAnalysis"),
  audioFeatures: jsonb("audioFeatures"),
  status: text("status").default('completed'),
  usageMode: text("usageMode").default('personal'),
  assessmentType: text("assessmentType").default('cognitive'),
  createdAt: timestamp("createdAt", { withTimezone: true }).defaultNow(),
  updatedAt: timestamp("updatedAt", { withTimezone: true }).defaultNow(),
});

// New: contact_messages table to store messages from ContactSection
export const contactMessages = pgTable("contact_messages", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  email: text("email").notNull(),
  subject: text("subject"),
  category: text("category"),
  message: text("message").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
});