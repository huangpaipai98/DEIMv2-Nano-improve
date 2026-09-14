import argparse
import tensorrt as trt
from pathlib import Path


def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        name = input_tensor.name
        shape = list(input_tensor.shape)
        
        min_shape = []
        opt_shape = []
        max_shape = []
        
        for dim in shape:
            if dim == -1:  # 动态维度
                min_shape.append(1)
                opt_shape.append(1)
                max_shape.append(1)
            else:
                min_shape.append(dim)
                opt_shape.append(dim)
                max_shape.append(dim)
        
        profile.set_shape(name, min_shape, opt_shape, max_shape)

    # 配置
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"[{Path(onnx_path).name}] FP16 mode enabled")
    else:
        print(f"[{Path(onnx_path).name}] Using FP32")

    # 构建
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build engine: {onnx_path}")

    # 保存
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"✓ Engine saved: {engine_path}")


def main(args):
    input_path = Path(args.onnx_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # 判断是单个文件还是文件夹
    if input_path.is_file() and input_path.suffix.lower() == ".onnx":
        # 单个文件
        engine_path = input_path.with_suffix(".engine")
        build_engine(str(input_path), str(engine_path), fp16=args.fp16)

    elif input_path.is_dir():
        # 文件夹：批量转换
        onnx_files = sorted(input_path.glob("*.onnx"))
        if not onnx_files:
            print(f"No .onnx files found in {input_path}")
            return

        for onnx_file in onnx_files:
            engine_path = onnx_file.with_suffix(".engine")
            try:
                build_engine(str(onnx_file), str(engine_path), fp16=args.fp16)
            except Exception as e:
                print(f"✗ Failed: {onnx_file.name}, error: {e}")
                # 继续转换下一个，不退出

    else:
        raise ValueError(f"Invalid path (not a .onnx file or directory): {input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', '-p', type=str,
                        help='Path to ONNX file or directory containing ONNX files')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    args = parser.parse_args()
    main(args)