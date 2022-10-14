version=$1

if [ "$version" = "2019" ]; then
    echo "SHELL : 2019 Vivado is selected"
    dnn_vivado='/opt/Xilinx/Vivado/2019.2/bin/vivado'
    dnn_hls='/opt/Xilinx/Vivado/2019.2/bin/vivado_hls'
else
  echo "SHELL : 2020 Vivado is selected"
  dnn_vivado='/opt/Xilinx/Vivado/2020.2/bin/vivado'
  dnn_hls='/opt/Xilinx/Vitis_HLS/2020.2/bin/vitis_hls'
fi