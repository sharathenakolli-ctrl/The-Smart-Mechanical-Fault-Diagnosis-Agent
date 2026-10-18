$wdir  = "C:\Users\SarathTeja\Desktop\The Smart Mechanical Fault Diagnosis Agent"
$pyExe = "C:\Users\SarathTeja\AppData\Local\Programs\Python\Python311\python.exe"
$log   = "$wdir\server.log"

"" | Set-Content $log   # clear log

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName               = $pyExe
$psi.Arguments              = "app.py"
$psi.WorkingDirectory       = $wdir
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute        = $false
$psi.CreateNoWindow         = $true
$psi.EnvironmentVariables["PYTHONUTF8"] = "1"

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

# Async event handlers so stdout/stderr don't block
$proc.EnableRaisingEvents = $true
$proc | Add-Member -NotePropertyName LogFile -NotePropertyValue $log

Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action {
    if ($Event.SourceEventArgs.Data) { Add-Content -Path $Event.MessageData -Value $Event.SourceEventArgs.Data }
} -MessageData $log | Out-Null

Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action {
    if ($Event.SourceEventArgs.Data) { Add-Content -Path $Event.MessageData -Value $Event.SourceEventArgs.Data }
} -MessageData $log | Out-Null

$proc.Start() | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

Write-Host "Server PID: $($proc.Id)"
$proc.Id | Set-Content "$wdir\server.pid"

# Wait up to 30 seconds for Flask to be ready
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    $content = Get-Content $log -ErrorAction SilentlyContinue
    if ($content -match "Running on http") {
        $ready = $true
        Write-Host "Flask server is UP."
        break
    }
    if (-not $proc.HasExited) {
        Write-Host "  waiting... ($i s)"
    } else {
        Write-Host "Process exited with code $($proc.ExitCode)"
        Write-Host "--- LOG ---"
        Get-Content $log
        break
    }
}

if ($ready) {
    Write-Host "--- SERVER LOG ---"
    Get-Content $log
}
