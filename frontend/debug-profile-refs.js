// Debug profile references
const fs = require('fs');
const path = require('path');

function scanDir(dir) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    if (fs.statSync(fullPath).isDirectory()) {
      scanDir(fullPath);
    } else if (file === 'route.ts') {
      const content = fs.readFileSync(fullPath, 'utf8');
      if (content.includes('profile') && content.includes('user.')) {
        console.log(`Found in ${fullPath}:`);
        const lines = content.split('\n');
        lines.forEach((line, index) => {
          if (line.includes('profile') && line.includes('user.')) {
            console.log(`  Line ${index + 1}: ${line.trim()}`);
          }
        });
        console.log('');
      }
    }
  }
}

console.log('🔍 DEBUGGING PROFILE REFERENCES:');
scanDir('./app/api');
