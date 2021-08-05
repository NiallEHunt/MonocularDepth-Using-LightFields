param(
  [switch]$IsRefocused = $false
)

Set-Location ..\..\models\monodepth2
conda activate monodepth2
if (-Not $IsRefocused) {
  python .\predict_hci.py --image_path ..\..\..\Data\HCI_benchmark\initial_test\ --output_path ..\..\..\Results\depth_maps\AiF\monodepth2\ --model_name mono+stereo_640x192 --ext png
}
else {
  python .\predict_hci.py --image_path ..\..\..\Data\HCI_benchmark\initial_test_refocused\ --output_path ..\..\..\Results\depth_maps\Refocused\monodepth2\ --model_name mono+stereo_640x192 --ext png
}
conda deactivate
Set-Location ..\..\evaluation\comparison-and-evaluation