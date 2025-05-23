# Convert LoRA to GGUF

This PowerShell script converts a LoRA adapter to GGUF format for use with Ollama.

param(
    [Parameter(Mandatory = $true)]
    [string]$LoraDir,
    
    [Parameter(Mandatory = $false)]
    [string]$GgufDir = $null
)

if (-not $GgufDir) {
    $GgufDir = Join-Path -Path $LoraDir -ChildPath "gguf"
}

# Run the conversion script
python convert_to_gguf.py --lora-dir $LoraDir --gguf-dir $GgufDir
