Option Explicit

Dim shell, fso, quote, scriptDirectory, batchPath
Dim localAppData, appDirectory, logDirectory, logPath, command

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
quote = Chr(34)

scriptDirectory = fso.GetParentFolderName(WScript.ScriptFullName)
batchPath = fso.BuildPath(scriptDirectory, "start-xiaomachi-wsl.bat")
localAppData = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%")
appDirectory = fso.BuildPath(localAppData, "Xiaomachi")
logDirectory = fso.BuildPath(appDirectory, "logs")
logPath = fso.BuildPath(logDirectory, "manual-start.log")

If Not fso.FolderExists(appDirectory) Then
    fso.CreateFolder appDirectory
End If
If Not fso.FolderExists(logDirectory) Then
    fso.CreateFolder logDirectory
End If

command = quote & shell.ExpandEnvironmentStrings("%ComSpec%") & quote _
    & " /d /c call " & quote & batchPath & quote & " --hidden >> " _
    & quote & logPath & quote & " 2>&1"
shell.Run command, 0, False
