param(
  [switch]$IsRefocused = $false
)

Set-Location ..\..\models\DenseDepth
conda activate DenseDepth
if (-Not $IsRefocused) {
  python .\test.py --input ..\..\..\Data\HCI_benchmark\initial_test\*.png --output ..\..\..\Results\depth_maps\AiF\DenseDepth
}
else {
  python .\test.py --input ..\..\..\Data\HCI_benchmark\initial_test_refocused\*.png --output ..\..\..\Results\depth_maps\Refocused\DenseDepth
}
conda deactivate
Set-Location ..\..\evaluation\comparison-and-evaluation