Param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Action
)

python src/main.py --action $Action