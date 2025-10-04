# Git Submodule Cleanup Instructions

Chạy các lệnh sau trong terminal để xóa hoàn toàn submodule:
```bash
# 1. Deinitialize submodule
git submodule deinit -f backend

# 2. Remove from git
git rm -f backend

# 3. Remove from .git/modules
rm -rf .git/modules/backend

# 4. Remove .gitmodules if empty
rm -f .gitmodules

# 5. Remove git config
git config --remove-section submodule.backend 2>/dev/null || true

# 6. Commit changes
git add .
git commit -m "Remove backend submodule completely"

# 7. Push to remote
git push origin main
```

