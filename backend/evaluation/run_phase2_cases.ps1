param(
    [string]$Dataset = "$PSScriptRoot\datasets\phase2_candidates_v1.json",
    [string]$BaseUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$utf8 = New-Object System.Text.UTF8Encoding($false)
$datasetText = $utf8.GetString([System.IO.File]::ReadAllBytes($Dataset))
$datasetObject = $datasetText | ConvertFrom-Json
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$resultDirectory = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Path $resultDirectory -Force | Out-Null
$results = @()

foreach ($case in $datasetObject.cases) {
    $body = @{ message = $case.user_question; user_id = "phase2-$runId" } | ConvertTo-Json -Compress
    try {
        $actual = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method Post `
            -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 180
        $trace = $null
        if ($actual.request_id) {
            try { $trace = Invoke-RestMethod -Uri "$BaseUrl/trace/tool/$($actual.request_id)" -TimeoutSec 15 }
            catch { $trace = @{ trace_error = $_.Exception.Message } }
        }
        $results += [ordered]@{
            case_id = $case.case_id
            question = $case.user_question
            expected_intent = $case.expected_intent
            gold_chunk = $case.gold_chunk
            expected_agent = $case.primary_agent
            expected_escalate = $case.should_escalate
            request_id = $actual.request_id
            conv_id = $actual.conv_id
            actual_intent = $actual.intent
            actual_agent = $actual.agent_type
            knowledge_used = $actual.knowledge_used
            tools_used = $actual.tools_used
            escalated = $actual.escalated
            latency_ms = $actual.latency_ms
            response = $actual.response
            trace = $trace
        }
        Write-Host "$($case.case_id) completed"
    } catch {
        $results += [ordered]@{ case_id = $case.case_id; question = $case.user_question; status = "request_failed"; error = $_.Exception.Message }
        Write-Warning "$($case.case_id) failed: $($_.Exception.Message)"
    }
}

$output = [ordered]@{
    run_id = $runId
    dataset = $Dataset
    base_url = $BaseUrl
    focus = @("intent_identification", "rag_recall_precision")
    results = $results
}
$outputPath = Join-Path $resultDirectory "phase2_run_$runId.json"
$output | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $outputPath -Encoding UTF8
Write-Host "Saved: $outputPath"
