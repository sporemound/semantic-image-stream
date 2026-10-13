# Reproduce original synthetic speech fixtures using a locally installed voice.
# Run with Windows PowerShell 5.1: powershell.exe -NoProfile -File .\scripts\create-validation-recordings.ps1
[CmdletBinding()]
param(
    [string]$Destination = '',
    [string]$Voice = '',
    [ValidateRange(-5, 5)][int]$Rate = 0
)
$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Destination) { $Destination = Join-Path $scriptDirectory '..\tests\fixtures' }
Add-Type -AssemblyName System.Speech
$destinationPath = [IO.Path]::GetFullPath($Destination)
[IO.Directory]::CreateDirectory($destinationPath) | Out-Null
$manifestPath = Join-Path $scriptDirectory '..\tests\fixtures\expected.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    if ($Voice) {
        $synth.SelectVoice($Voice)
    } else {
        $englishVoice = $synth.GetInstalledVoices() | Where-Object {
            $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*'
        } | Select-Object -First 1
        if (-not $englishVoice) { throw 'Install an English Windows speech voice before generating fixtures.' }
        $synth.SelectVoice($englishVoice.VoiceInfo.Name)
    }
    $synth.Rate = $Rate
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
        16000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono
    )
    foreach ($recording in $manifest.recordings) {
        $outputPath = Join-Path $destinationPath $recording.file
        $synth.SetOutputToWaveFile($outputPath, $format)
        $prompt = New-Object System.Speech.Synthesis.PromptBuilder([Globalization.CultureInfo]'en-US')
        if ($recording.parts) {
            for ($index = 0; $index -lt $recording.parts.Count; $index++) {
                if ($index -gt 0) { $prompt.AppendBreak([TimeSpan]::FromSeconds($recording.break_seconds)) }
                $prompt.AppendText($recording.parts[$index])
            }
        } else {
            $prompt.AppendText($recording.text)
        }
        $synth.Speak($prompt)
        $synth.SetOutputToNull()
        Write-Output ('Created {0} using {1}' -f $outputPath, $synth.Voice.Name)
    }
} finally {
    $synth.Dispose()
}
