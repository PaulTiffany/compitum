Param(
  [switch]$Commit = $false,
  [string]$Message = "Update wiki with generated notebook embeds"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Running embed script..." -ForegroundColor Cyan
python scripts/embed_notebooks_in_wiki.py

Write-Host "[2/3] Wiki status:" -ForegroundColor Cyan
git -C compitum.wiki status --porcelain=v1

if ($Commit) {
  Write-Host "[3/3] Committing and pushing wiki changes..." -ForegroundColor Cyan
  git -C compitum.wiki add -A
  git -C compitum.wiki commit -m $Message
  git -C compitum.wiki push
} else {
  Write-Host "[3/3] Review changes in compitum.wiki, then commit/push manually:" -ForegroundColor Yellow
  Write-Host "  cd compitum.wiki" -ForegroundColor DarkGray
  Write-Host "  git add -A" -ForegroundColor DarkGray
  Write-Host "  git commit -m 'Update wiki with generated notebook embeds'" -ForegroundColor DarkGray
  Write-Host "  git push" -ForegroundColor DarkGray
}

