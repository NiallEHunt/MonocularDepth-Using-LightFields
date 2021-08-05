param(
  [switch]$IsRefocused = $false
)

Set-Location ..\..\models\defocus-net
conda activate defocus-net
if (-Not $IsRefocused) {
  python .\predict_depth.py --image_path ..\..\..\Data\HCI_benchmark\initial_test\ --output_path ..\..\..\Results\depth_maps\AiF\defocus-net\  --final_model True --multiple_focus True
}
else {
  python .\predict_depth.py --image_path ..\..\..\Data\HCI_benchmark\initial_test_refocused\ --output_path ..\..\..\Results\depth_maps\Refocused\defocus-net\  --final_model True --multiple_focus True
}
conda deactivate
Set-Location ..\..\evaluation\comparison-and-evaluation