param(
    [switch]$OnlyWhenLoginRequired
)

$ErrorActionPreference = "Stop"

function Invoke-LocalJsonPost {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Json,
        [string]$BearerToken = ""
    )

    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(8)
    try {
        if (-not [string]::IsNullOrWhiteSpace($BearerToken)) {
            $client.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", $BearerToken)
        }
        $content = New-Object System.Net.Http.StringContent($Json, [Text.Encoding]::UTF8, "application/json")
        $response = $client.PostAsync($Uri, $content).GetAwaiter().GetResult()
        if ([int]$response.StatusCode -ge 300 -and [int]$response.StatusCode -lt 400) {
            throw "NapCat WebUI returned a redirect."
        }
        $response.EnsureSuccessStatusCode()
        return $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
    }
    finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

function Test-NapCatLogin {
    param([Parameter(Mandatory = $true)][string]$Token)

    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($Token + ".napcat"))
        $hash = -join ($hashBytes | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $sha256.Dispose()
    }

    $loginJson = @{ hash = $hash } | ConvertTo-Json -Compress
    $auth = Invoke-LocalJsonPost -Uri "http://127.0.0.1:6099/api/auth/login" -Json $loginJson
    $credential = [string]$auth.data.Credential
    if ([string]::IsNullOrWhiteSpace($credential)) {
        return $false
    }

    $status = Invoke-LocalJsonPost `
        -Uri "http://127.0.0.1:6099/api/QQLogin/CheckLoginStatus" `
        -Json "{}" `
        -BearerToken $credential
    return $status.data.isLogin -eq $true
}

try {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
    $configPath = Join-Path $repoRoot "infra\wsl\runtime\napcat\config\webui.json"
    $config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
    $token = [string]$config.token
    if ([string]::IsNullOrWhiteSpace($token)) {
        exit 1
    }

    if ($OnlyWhenLoginRequired) {
        try {
            if (Test-NapCatLogin -Token $token) {
                exit 0
            }
        }
        catch {
            # An unknown status is treated as requiring attention.
        }
    }

    $encodedToken = [uri]::EscapeDataString($token)
    $url = "http://127.0.0.1:6099/webui/qq_login?token=$encodedToken"
    Start-Process -FilePath $url
}
catch {
    exit 1
}
