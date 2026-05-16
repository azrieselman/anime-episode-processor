; Inno Setup script for the Anime Episode Processor Windows installer.
;
; Build:    iscc packaging\installer.iss
; Output:   dist\AEP-Setup-1.0.0-beta3.exe
;
; Prerequisites:
;   * `pyinstaller packaging/aep.spec --clean` has produced dist\aep-gui\.
;   * Inno Setup 6.x is installed (https://jrsoftware.org/isinfo.php) and
;     `iscc` is reachable on PATH (or invoked by absolute path).
;
; The AppId is a fixed GUID per Inno Setup convention — DO NOT regenerate it
; for new builds, otherwise upgrades from older installs would re-install
; alongside the existing one instead of replacing it.

#define MyAppName        "Anime Episode Processor"
#define MyAppShortName   "AEP"
#define MyAppVersion     "1.0.0-beta3"
#define MyAppPublisher   "Andreas Rieselman"
#define MyAppURL         "https://github.com/azrieselman/anime-episode-processor"
#define MyAppExeName     "aep-gui.exe"

[Setup]
AppId={{7B4E5C8A-4D0F-4F61-9A2E-9B8C3FB1D7AA}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\AnimeEpisodeProcessor
DefaultGroupName={#MyAppShortName}
DisableProgramGroupPage=auto
LicenseFile=..\LICENSE
InfoBeforeFile=..\THIRD_PARTY_NOTICES.md
OutputDir=..\dist
OutputBaseFilename=AEP-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
SetupIconFile=..\src\aep\gui\resources\app.ico
WizardImageStretch=no
DisableWelcomePage=no
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Pull in the entire one-folder PyInstaller bundle. The wildcard recurses
; into subdirectories so Qt platform plugins, presets/, pipelines/, etc. all
; ship together.
Source: "..\dist\aep-gui\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Top-level documentation that should be visible in the install dir without
; digging into the data\ subtree. PyInstaller already places these into
; dist\aep-gui\ via aep.spec's `datas`, so the next two entries are mostly
; insurance for users who delete files post-install.
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up runtime state under %LOCALAPPDATA%\AEP\AnimeEpisodeProcessor.
; This tree contains fetched tools plus logs/cache/jobs/presets created after
; install, so it is not covered by [Files].
Type: filesandordirs; Name: "{localappdata}\AEP\AnimeEpisodeProcessor"
; Back-compat cleanup for older betas that wrote under legacy locations.
Type: filesandordirs; Name: "{localappdata}\AEP\tools"
Type: filesandordirs; Name: "{localappdata}\AEP\runtime"
