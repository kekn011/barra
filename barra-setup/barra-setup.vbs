' barra-setup.vbs — DOPPELKLICK: startet barra Setup ohne Konsolenfenster (kein WSL, kein Admin).
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & dir & "\barra-setup.ps1""", 0, False
