Param(
    [string]$Message = "Initial commit (reset history)"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git($Args) {
    Write-Host "> git $Args"
    git $Args
}

Write-Host "--- Resetting git history (orphan) ---" -ForegroundColor Cyan

# 1) Create orphan branch without history
Invoke-Git "checkout --orphan latest_branch"

# 2) Stage all files according to current .gitignore
Invoke-Git "add -A"

# 3) Single commit
Invoke-Git "commit -m `"$Message`""

# 4) Delete old main if exists (ignore error)
try { Invoke-Git "branch -D main" } catch { Write-Host "(info) main not found, continue" -ForegroundColor Yellow }

# 5) Rename current branch to main
Invoke-Git "branch -m main"

# 6) Force push to origin
Invoke-Git "push -f origin main"

Write-Host "Done." -ForegroundColor Green

