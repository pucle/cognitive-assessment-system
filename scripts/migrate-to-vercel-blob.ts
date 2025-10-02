#!/usr/bin/env tsx
/**
 * Migration script to upload existing local audio files to Vercel Blob Storage
 * Run with: npx tsx scripts/migrate-to-vercel-blob.ts
 */

import { put } from '@vercel/blob';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Load environment variables
import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface MigrationResult {
  success: number;
  failed: number;
  errors: string[];
}

async function getFilesRecursively(dir: string): Promise<string[]> {
  const files: string[] = [];
  
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory()) {
        const subFiles = await getFilesRecursively(fullPath);
        files.push(...subFiles);
      } else if (entry.isFile() && isAudioFile(entry.name)) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    console.warn(`Cannot read directory ${dir}:`, error instanceof Error ? error.message : error);
  }
  
  return files;
}

function isAudioFile(filename: string): boolean {
  const audioExtensions = ['.webm', '.wav', '.mp3', '.mp4', '.ogg', '.m4a'];
  return audioExtensions.some(ext => filename.toLowerCase().endsWith(ext));
}

function generateBlobPath(localPath: string, baseDir: string): string {
  // Convert local path to blob storage path
  const relativePath = path.relative(baseDir, localPath);
  const normalizedPath = relativePath.replace(/\\/g, '/'); // Convert Windows paths
  
  // Add timestamp to avoid collisions
  const timestamp = Date.now();
  const dir = path.dirname(normalizedPath);
  const filename = path.basename(normalizedPath);
  
  return `migrated/${dir}/${timestamp}_${filename}`.replace(/\/+/g, '/');
}

async function uploadFileToBlob(localPath: string, blobPath: string): Promise<{ url: string }> {
  try {
    const fileBuffer = await fs.readFile(localPath);
    const stats = await fs.stat(localPath);
    
    console.log(`📤 Uploading: ${localPath} -> ${blobPath} (${stats.size} bytes)`);
    
    const blob = await put(blobPath, fileBuffer, {
      access: 'public',
      addRandomSuffix: false,
    });
    
    return blob;
  } catch (error) {
    throw new Error(`Failed to upload ${localPath}: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function migrateDirectory(dirPath: string): Promise<MigrationResult> {
  const result: MigrationResult = {
    success: 0,
    failed: 0,
    errors: []
  };

  console.log(`🔍 Scanning directory: ${dirPath}`);
  
  try {
    const files = await getFilesRecursively(dirPath);
    console.log(`📁 Found ${files.length} audio files to migrate`);
    
    if (files.length === 0) {
      console.log('ℹ️  No audio files found to migrate');
      return result;
    }

    for (const localPath of files) {
      try {
        const blobPath = generateBlobPath(localPath, dirPath);
        await uploadFileToBlob(localPath, blobPath);
        
        result.success++;
        console.log(`✅ Successfully uploaded: ${path.basename(localPath)}`);
        
        // Small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        result.failed++;
        const errorMsg = `Failed to upload ${localPath}: ${error instanceof Error ? error.message : 'Unknown error'}`;
        result.errors.push(errorMsg);
        console.error(`❌ ${errorMsg}`);
      }
    }
  } catch (error) {
    const errorMsg = `Failed to scan directory ${dirPath}: ${error instanceof Error ? error.message : 'Unknown error'}`;
    result.errors.push(errorMsg);
    console.error(`❌ ${errorMsg}`);
  }

  return result;
}

async function main() {
  console.log('🚀 Starting migration to Vercel Blob Storage...');
  
  // Check if BLOB_READ_WRITE_TOKEN is set
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('❌ BLOB_READ_WRITE_TOKEN environment variable is not set');
    console.log('Please add BLOB_READ_WRITE_TOKEN to your .env.local file');
    process.exit(1);
  }

  const baseDir = path.resolve(__dirname, '..');
  const directoriesToMigrate = [
    path.join(baseDir, 'recordings'),
    path.join(baseDir, 'records'),
  ];

  let totalSuccess = 0;
  let totalFailed = 0;
  const allErrors: string[] = [];

  for (const dir of directoriesToMigrate) {
    console.log(`\n📂 Migrating directory: ${dir}`);
    
    try {
      await fs.access(dir);
      const result = await migrateDirectory(dir);
      
      totalSuccess += result.success;
      totalFailed += result.failed;
      allErrors.push(...result.errors);
      
      console.log(`📊 Directory ${path.basename(dir)} results:`);
      console.log(`   ✅ Success: ${result.success}`);
      console.log(`   ❌ Failed: ${result.failed}`);
      
    } catch (error) {
      console.log(`⚠️  Directory ${dir} does not exist or is not accessible`);
    }
  }

  console.log(`\n🎯 Migration Summary:`);
  console.log(`   📁 Total files successfully migrated: ${totalSuccess}`);
  console.log(`   ❌ Total files failed: ${totalFailed}`);
  
  if (allErrors.length > 0) {
    console.log(`\n🔍 Errors encountered:`);
    allErrors.forEach((error, index) => {
      console.log(`   ${index + 1}. ${error}`);
    });
  }
  
  if (totalFailed === 0) {
    console.log(`\n🎉 All files migrated successfully!`);
  } else {
    console.log(`\n⚠️  Migration completed with ${totalFailed} errors. Check the logs above.`);
  }
}

// Run the migration
main().catch((error) => {
  console.error('💥 Migration failed:', error);
  process.exit(1);
});

export { };
