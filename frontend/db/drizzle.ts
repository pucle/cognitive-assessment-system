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
// Note: fetchConnectionCache is enabled by default in newer versions.
// The following options are no longer supported and have been removed.
// neonConfig.poolSize = 10;
// neonConfig.queueLimit = 100;

const sql = neon(databaseUrl);
const db = drizzle(sql, { schema });

export default db;

