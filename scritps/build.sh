#python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --cuda_architectures "100-real" --dist_dir ./dist-dir --build_type Release --use_ccache #--clean
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --cuda_architectures "90-real;100-real" --dist_dir ./dist-dir --build_type Release --use_ccache #--clean
