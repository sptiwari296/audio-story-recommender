# Captures demo screenshots of the running Streamlit app (start it first: streamlit run app.py)
$edge = @("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "C:\Program Files\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$out = Join-Path $PSScriptRoot "screenshots"
New-Item -ItemType Directory -Force $out | Out-Null
$shots = @{ "01_recommendations.png" = "http://localhost:8501/"; "02_cold_start.png" = "http://localhost:8501/?mode=new" }
foreach ($name in $shots.Keys) {
    & $edge --headless=new --disable-gpu --hide-scrollbars --window-size=1400,2200 --virtual-time-budget=25000 `
        --screenshot="$out\$name" $shots[$name] | Out-Null
}
Copy-Item (Join-Path $PSScriptRoot "eval\metrics.png") "$out\03_evaluation_chart.png" -Force
Get-ChildItem $out | Select-Object Name, Length
