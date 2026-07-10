param(
    [ValidateSet("onebot_still_offline", "onebot_session_unhealthy", "webui_login_error", "napcat_restart_failed")]
    [string]$Reason = "onebot_still_offline"
)

$title = "Xiaomachi WSL alert"
$message = switch ($Reason) {
    "napcat_restart_failed" { "NapCat could not be restarted. Run status-xiaomachi-wsl.bat and check the logs." }
    "webui_login_error" { "NapCat reports a QQ login error. Open the NapCat WebUI and sign in again." }
    "onebot_session_unhealthy" { "Xiaomachi QQ did not recover after an automatic NapCat restart. Open the NapCat WebUI and check the QQ login." }
    default { "Xiaomachi QQ is still offline after an automatic NapCat restart. Open the NapCat WebUI and sign in again." }
}

Add-Type -AssemblyName System.Windows.Forms
[void][System.Windows.Forms.MessageBox]::Show(
    $message,
    $title,
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Warning,
    [System.Windows.Forms.MessageBoxDefaultButton]::Button1,
    [System.Windows.Forms.MessageBoxOptions]::ServiceNotification
)
