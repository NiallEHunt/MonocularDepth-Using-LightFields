param(
  [switch]$IsRefocused = $false
)

Set-Location ..\..\models\GDN-Pytorch\src
conda activate Old-GDN
if (-Not $IsRefocused) {
  python .\depth_extract.py --model_dir ..\models\GDN_RtoD_pretrained.pkl --output_path ..\..\..\..\Results\depth_maps\AiF\GDN-Pytorch\ --input_path ..\..\..\..\Data\HCI_benchmark\initial_test\
}
else {
  python .\depth_extract.py --model_dir ..\models\GDN_RtoD_pretrained.pkl --output_path ..\..\..\..\Results\depth_maps\Refocused\GDN-Pytorch\ --input_path ..\..\..\..\Data\HCI_benchmark\initial_test_refocused\
}
conda deactivate
Set-Location ..\..\..\evaluation\comparison-and-evaluation