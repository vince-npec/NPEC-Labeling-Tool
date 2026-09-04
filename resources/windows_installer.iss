#define MyAppName "NPEC Labeling Tool"
#define MyAppPublisher "NPEC / Utrecht University"
#define MyAppExeName "NPEC Labeling Tool.exe"

#ifndef MyAppVersion
  #define MyAppVersion "2026.02"
#endif

#ifndef MySourceDir
  #define MySourceDir "dist-win\NPEC Labeling Tool"
#endif

#ifndef MyOutputDir
  #define MyOutputDir "dist-win"
#endif

[Setup]
AppId={{8364F3FB-3739-45BC-B676-4A9577770F22}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir={#MyOutputDir}
OutputBaseFilename=NPEC-Labeling-Tool-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
