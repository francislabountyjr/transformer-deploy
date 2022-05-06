import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from transformer_deploy.backends.trt_utils import build_engine
import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding argumenta
parser.add_argument("-input", "--Input", help="Input .onnx file location")
parser.add_argument("-output", "--Output", help="Output .plan file location")

# Read arguments from command line
args = parser.parse_args()

if args.Input and args.Output:

    trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
    runtime: Runtime = trt.Runtime(trt_logger)
    profile_index = 0
    max_seq_len = 2048
    batch_size = 4

    engine = build_engine(
        runtime=runtime,
        onnx_file_path=args.Input,
        logger=trt_logger,
        min_shape=(1, max_seq_len),
        optimal_shape=(batch_size, max_seq_len*0.5),
        max_shape=(batch_size, max_seq_len),
        workspace_size=100000 * 1024 * 1024 * 20,
        fp16=True,
        int8=False,
    )

    with open(args.Output, 'wb') as f:
        f.write(engine.serialize())

else:
    print("Please supply '-input' and '-output' arguments")
