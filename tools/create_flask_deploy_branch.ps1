# PowerShell helper: create a deploy branch for pure-Flask deployment
# Usage: run from repository root (PowerShell)

param(
    [string]$branchName = 'deploy/flask',
    [switch]$push
)

Write-Host "Creating deploy branch: $branchName"

git checkout -b $branchName

Write-Host "Removing non-Flask files from branch (these files will be deleted from the branch)"
# Files considered unnecessary for pure Flask deploy
$remove = @(
    'web_server.py',
    'requirements_web.txt',
    'app.py',
    'camera_widget.py',
    'main_window.py'
)

foreach ($f in $remove) {
    if (Test-Path $f) {
        git rm -f $f -q
        Write-Host "git rm $f"
    }
}

Write-Host "Add .gitignore (already present) and commit"
git add .gitignore Dockerfile render.yaml requirements.txt README_deploy.md
git commit -m "Prepare deploy branch: strip non-Flask files, update Docker/render configs" -q

if ($push) {
    git push -u origin $branchName
    Write-Host "Pushed branch to origin/$branchName"
} else {
    Write-Host "Branch created locally: $branchName. Use -push to push to origin."
}

Write-Host "Done. Review the branch, run tests locally, then push and create Render service pointing to this branch."
