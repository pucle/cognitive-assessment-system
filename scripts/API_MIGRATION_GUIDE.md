# API Routes Migration Guide

## Existing Routes Found:
- app\api\save-recording-blob (app\api\save-recording-blob\route.ts)
- app\api\upload-audio-blob (app\api\upload-audio-blob\route.ts)

## Migration Steps:
1. Convert Next.js API routes to Express.js routes
2. Update imports and exports
3. Adapt request/response handling
4. Move to backend/src/routes/

## Example Conversion:

### Before (Next.js):
```typescript
// app/api/users/route.ts
import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  return NextResponse.json({ users: [] })
}
```

### After (Express.js):
```javascript
// backend/src/routes/users.js
const express = require('express')
const router = express.Router()

router.get('/', async (req, res) => {
  res.json({ users: [] })
})

module.exports = router
```
