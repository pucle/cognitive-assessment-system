import 'server-only';
import { drizzle } from 'drizzle-orm/neon-http';
import { neon, neonConfig } from "@neondatabase/serverless"
import * as schema from "./schema"

// Validate DATABASE_URL format
const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error('DATABASE_URL is not defined');
}

if (!databaseUrl.startsWith('postgresql://')) {
  throw new Error('DATABASE_URL must start with postgresql://');
}

// Configure connection pooling for better performance
// Note: fetchConnectionCache is now always true by default in newer versions
neonConfig.poolSize = 10; // Connection pool size
neonConfig.queueLimit = 100; // Queue limit for pending connections

const sql = neon(databaseUrl);
const db = drizzle(sql, { schema });

export default db;

