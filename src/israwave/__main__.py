import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
import soundfile as sf
import time
from nakdimon_ort import Nakdimon

from israwave import OptiSpeechONNXModel

from .values import InferenceInputs, InferenceOutputs

log = logging.getLogger("infer")
ONNX_CUDA_PROVIDERS = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
ONNX_CPU_PROVIDERS = [
    "CPUExecutionProvider",
]


def main():
    logging.basicConfig()
    
    parser = argparse.ArgumentParser(description=" ONNX inference of OptiSpeech")

    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the exported OptiSpeech ONNX model",
    )
    parser.add_argument("text", type=str, help="Text to speak")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write generated audio to.",
    )
    parser.add_argument("--d-factor", type=float, default=1.0, help="Scale to control speech rate.")
    parser.add_argument("--p-factor", type=float, default=1.0, help="Scale to control pitch.")
    parser.add_argument("--e-factor", type=float, default=1.0, help="Scale to control energy.")
    parser.add_argument("--no-split", action="store_true", help="Don't split input text into sentences.")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load model
    onnx_providers = ONNX_CUDA_PROVIDERS if args.cuda else ONNX_CPU_PROVIDERS
    model = OptiSpeechONNXModel.from_onnx_file_path(args.onnx_path, onnx_providers=onnx_providers)
    diacritics_model = Nakdimon('nakdimon.onnx')

    # Process text
    args.text = diacritics_model.compute(args.text)
    inputs = model.prepare_input(
        args.text,
        d_factor=args.d_factor,
        p_factor=args.p_factor,
        e_factor=args.e_factor,
        split_sentences=not args.no_split,
        lang='he'
    )
    log.info(f"Normalized text: {inputs.clean_text}")
    # Perform inference
    start = time.time()
    outputs = model.synthesise(inputs)
    print(time.time() - start)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, wav in enumerate(outputs.unbatched_wavs()):
        outfile = output_dir.joinpath(f"gen-{i + 1}")
        out_wav = outfile.with_suffix(".wav")
        wav = wav.squeeze()
        sf.write(out_wav, wav, model.sample_rate)
        log.info(f"Wrote wav to: `{out_wav}`")

    latency = outputs.latency
    rtf = outputs.rtf
    log.info(f"OptiSpeech latency: {round(latency)} ms")
    log.info(f"OptiSpeech RTF: {rtf}")


if __name__ == "__main__":
    main()
