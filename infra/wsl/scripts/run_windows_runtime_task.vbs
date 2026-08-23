Option Explicit

Dim shell, fso, regex, quote, distro, windowsDirectory, wslPath
Dim localAppData, appDirectory, logDirectory, logPath
Dim command, exitCode, logFile, startedAt, runtimeSeconds
Dim quickExitCount, restartDelayMilliseconds

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
Set regex = New RegExp
quote = Chr(34)
distro = "Ubuntu_Migrated"

If WScript.Arguments.Count > 0 Then
    distro = WScript.Arguments(0)
End If
regex.Pattern = "^[A-Za-z0-9._-]+$"
If Not regex.Test(distro) Then
    WScript.Quit 2
End If

windowsDirectory = shell.ExpandEnvironmentStrings("%WINDIR%")
wslPath = fso.BuildPath(fso.BuildPath(windowsDirectory, "System32"), "wsl.exe")
localAppData = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%")
appDirectory = fso.BuildPath(localAppData, "Xiaomachi")
logDirectory = fso.BuildPath(appDirectory, "logs")
logPath = fso.BuildPath(logDirectory, "wsl-runtime-task.log")

If Not fso.FolderExists(appDirectory) Then
    fso.CreateFolder appDirectory
End If
If Not fso.FolderExists(logDirectory) Then
    fso.CreateFolder logDirectory
End If

command = quote & wslPath & quote _
    & " -d " & distro _
    & " --user root --exec /usr/local/bin/xiaomachi-wsl-entry anchor"

quickExitCount = 0

Do
    startedAt = Now
    Set logFile = fso.OpenTextFile(logPath, 8, True)
    logFile.WriteLine startedAt & " starting distro=" & distro _
        & " launcher=wscript-direct"
    logFile.Close

    exitCode = shell.Run(command, 0, True)

    runtimeSeconds = DateDiff("s", startedAt, Now)
    If runtimeSeconds >= 5 Then
        quickExitCount = 0
        restartDelayMilliseconds = 200
    Else
        quickExitCount = quickExitCount + 1
        Select Case quickExitCount
            Case 1
                restartDelayMilliseconds = 200
            Case 2
                restartDelayMilliseconds = 1000
            Case 3
                restartDelayMilliseconds = 5000
            Case Else
                restartDelayMilliseconds = 15000
        End Select
    End If

    Set logFile = fso.OpenTextFile(logPath, 8, True)
    logFile.WriteLine Now & " anchor_stopped exit_code=" & exitCode _
        & " runtime_seconds=" & runtimeSeconds _
        & " quick_exit_count=" & quickExitCount _
        & " restarting_in_milliseconds=" & restartDelayMilliseconds
    logFile.Close
    WScript.Sleep restartDelayMilliseconds
Loop
