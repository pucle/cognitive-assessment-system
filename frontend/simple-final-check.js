// Simple Final Check - Database & API Status
const { neon } = require('@neondatabase/serverless');
const http = require('http');
require('dotenv').config({ path: '.env.local' });

const sql = neon(process.env.DATABASE_URL || process.env.NEON_DATABASE_URL || '');

async function simpleCheck() {
  console.log('🎯 SIMPLE FINAL SYSTEM CHECK\n');

  let passed = 0;
  let total = 0;

  // 1. Database Schema Check
  console.log('🔍 DATABASE SCHEMA CHECK:');
  try {
    const userColumns = await sql`SELECT column_name FROM information_schema.columns WHERE table_name = 'users' ORDER BY column_name`;
    const actualColumns = userColumns.map(c => c.column_name).sort();

    console.log(`   ✅ Found ${actualColumns.length} columns in users table`);
    console.log(`   Columns: ${actualColumns.join(', ')}`);

    // Check for profile column
    const hasProfile = actualColumns.includes('profile');
    if (!hasProfile) {
      console.log('   ✅ No problematic profile column');
      passed++;
    } else {
      console.log('   ❌ Profile column still exists');
    }
    total++;

    // Check data
    const userCount = await sql`SELECT COUNT(*) as count FROM users`;
    console.log(`   ✅ Users table has ${userCount[0].count} records`);
    passed++;
    total++;

  } catch (error) {
    console.log(`   ❌ Database error: ${error.message}`);
    total++;
  }

  // 2. API Endpoints Check
  console.log('\n🚀 API ENDPOINTS CHECK:');
  const endpoints = [
    'http://localhost:3000/api/database/user?userId=2',
    'http://localhost:3000/api/profile?email=test@example.com',
    'http://localhost:3000/api/community'
  ];

  function makeRequest(url) {
    return new Promise((resolve) => {
      http.get(url, { timeout: 5000 }, (res) => {
        let data = '';
        res.on('data', (chunk) => data += chunk);
        res.on('end', () => {
          try {
            const json = JSON.parse(data);
            resolve({ status: res.statusCode, success: json.success !== false });
          } catch (e) {
            resolve({ status: res.statusCode, success: false });
          }
        });
      }).on('error', () => resolve({ status: 0, success: false }))
        .on('timeout', () => resolve({ status: 0, success: false }));
    });
  }

  for (const url of endpoints) {
    try {
      const result = await makeRequest(url);
      if (result.status === 200 && result.success) {
        console.log(`   ✅ ${url.split('/').pop()} - OK`);
        passed++;
      } else {
        console.log(`   ❌ ${url.split('/').pop()} - Status: ${result.status}`);
      }
    } catch (error) {
      console.log(`   ❌ ${url.split('/').pop()} - Error`);
    }
    total++;
  }

  // 3. Code Quality Check
  console.log('\n💻 CODE QUALITY CHECK:');
  const fs = require('fs');

  try {
    // Check schema file - exclude commented lines
    const schemaContent = fs.readFileSync('./db/schema.ts', 'utf8');
    const activeLines = schemaContent.split('\n').filter(line =>
      line.trim() && !line.trim().startsWith('//')
    ).join('\n');
    const hasProfileColumn = activeLines.includes('profile: jsonb("profile")');

    if (!hasProfileColumn) {
      console.log('   ✅ Schema profile column properly commented out');
      passed++;
    } else {
      console.log('   ❌ Schema still has active profile column');
    }
    total++;

    // Check API routes for profile references
    let profileRefs = 0;
    const apiDir = './app/api';

    function scanDir(dir) {
      if (!fs.existsSync(dir)) return;
      const files = fs.readdirSync(dir);
      for (const file of files) {
        const fullPath = require('path').join(dir, file);
        if (fs.statSync(fullPath).isDirectory()) {
          scanDir(fullPath);
        } else if (file === 'route.ts') {
          const content = fs.readFileSync(fullPath, 'utf8');
          if (content.includes('profile') && content.includes('user.')) {
            profileRefs++;
          }
        }
      }
    }

    scanDir(apiDir);
    if (profileRefs === 0) {
      console.log('   ✅ No profile column references in API routes');
      passed++;
    } else {
      console.log(`   ❌ Found ${profileRefs} profile references in API routes`);
    }
    total++;

  } catch (error) {
    console.log(`   ❌ Code quality check error: ${error.message}`);
    total++;
  }

  // Final Summary
  console.log('\n📊 FINAL SUMMARY:');
  console.log(`   Total Checks: ${total}`);
  console.log(`   Passed: ${passed}`);
  console.log(`   Failed: ${total - passed}`);
  console.log(`   Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

  if (passed / total >= 0.8) {
    console.log('\n🟢 SYSTEM STATUS: PRODUCTION READY');
  } else {
    console.log('\n🔴 SYSTEM STATUS: NEEDS FIXES');
  }

  console.log('\n🎯 CHECK COMPLETE');
}

simpleCheck().catch(console.error);
