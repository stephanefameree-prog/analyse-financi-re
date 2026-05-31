# Publie le dashboard sur GitHub (première fois ou mise à jour).
# Usage :
#   .\publier_github.ps1 -RemoteUrl "https://github.com/MON_COMPTE/MON_REPO.git"
#   .\publier_github.ps1   # met à jour seulement si remote déjà configuré

param(
    [string]$RemoteUrl = "",
    [string]$CommitMessage = "Dashboard financier V6.3 — analyse portefeuille Streamlit"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Require-Git {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Host ""
        Write-Host "Git n'est pas installé ou pas dans le PATH." -ForegroundColor Red
        Write-Host "1. Téléchargez Git : https://git-scm.com/download/win"
        Write-Host "2. Relancez ce script dans PowerShell"
        Write-Host ""
        exit 1
    }
}

Require-Git

if (-not (git config user.email 2>$null) -or -not (git config user.name 2>$null)) {
    Write-Host ""
    Write-Host "Identité Git requise (une seule fois, pour ce dossier) :" -ForegroundColor Yellow
    Write-Host '  git config user.name "Votre Nom"'
    Write-Host '  git config user.email "votre@email.com"'
    Write-Host ""
    Write-Host "Utilisez l’email de votre compte GitHub (ou noreply@github.com)."
    exit 1
}

if (-not (Test-Path ".git")) {
    Write-Host "Initialisation du dépôt Git..." -ForegroundColor Cyan
    git init
    git branch -M main
}

if ($RemoteUrl -and -not (git remote get-url origin 2>$null)) {
    git remote add origin $RemoteUrl
    Write-Host "Remote origin : $RemoteUrl" -ForegroundColor Green
}

$blockedPatterns = @(
    "mon_portefeuille.csv",
    "watchlists.json",
    "dividendes_cache.json",
    "fundamentals_cache.json"
)
foreach ($pattern in $blockedPatterns) {
    if (Test-Path $pattern) {
        $tracked = git ls-files --error-unmatch $pattern 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "ERREUR : $pattern est suivi par Git — retrait du suivi requis." -ForegroundColor Red
            Write-Host "  git rm --cached $pattern"
            exit 1
        }
    }
}

git add -A

$staged = git diff --cached --name-only
foreach ($file in $staged) {
    foreach ($pattern in $blockedPatterns) {
        if ($file -eq $pattern -or $file -like "*\$pattern") {
            Write-Host "ERREUR : fichier sensible prêt à être envoyé : $file" -ForegroundColor Red
            Write-Host "Annulation — vérifiez .gitignore."
            git reset HEAD -- $file 2>$null
            exit 1
        }
    }
}
git status

$changes = git status --porcelain
if (-not $changes) {
    Write-Host "Rien à commiter — dépôt déjà à jour." -ForegroundColor Yellow
} else {
    git commit -m $CommitMessage
    Write-Host "Commit créé." -ForegroundColor Green
}

$hasOrigin = git remote get-url origin 2>$null
if (-not $hasOrigin) {
    Write-Host ""
    Write-Host "Étapes restantes :" -ForegroundColor Cyan
    Write-Host "1. Créez un repo VIDE sur https://github.com/new"
    Write-Host "2. Relancez :"
    Write-Host '   .\publier_github.ps1 -RemoteUrl "https://github.com/VOTRE_COMPTE/VOTRE_REPO.git"'
    exit 0
}

Write-Host "Envoi vers GitHub (origin main)..." -ForegroundColor Cyan
git push -u origin main
Write-Host "Terminé." -ForegroundColor Green
