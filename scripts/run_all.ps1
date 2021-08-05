param(
  [switch]$IsRefocused = $false
)

if ( -Not $IsRefocused ) {
  .\run_defocus-net.ps1 
  .\run_DenseDepth.ps1 
  .\run_gdn_pytorch.ps1
  .\run_monodepth.ps1
}
else {
  .\run_defocus-net.ps1 -IsRefocused
  .\run_DenseDepth.ps1 -IsRefocused
  .\run_gdn_pytorch.ps1 -IsRefocused
  .\run_monodepth.ps1 -IsRefocused
}