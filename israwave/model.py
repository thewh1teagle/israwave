import json
import logging
from time import perf_counter
import numpy as np
import onnxruntime
from israwave.tokenizer import IPATokenizer
from .tensors import InferenceInputs, InferenceOutputs

log = logging.getLogger("infer")

class Model:

    def __init__(self, onnx_path: str, espeak_data_path: str, onnx_providers: list[str] = ["CPUExecutionProvider"]):
        session = onnxruntime.InferenceSession(onnx_path, providers=onnx_providers)
        meta = session.get_modelmeta()
        infer_params = json.loads(meta.custom_metadata_map["inference"])
        self.tokenizer=IPATokenizer(espeak_data_path)
        self.session=session
        self.name=infer_params["name"]
        self.sample_rate=infer_params["sample_rate"]
        self.inference_args=infer_params["inference_args"]
        self.speakers=infer_params["speakers"]
        self.languages=infer_params["languages"]

    def __post_init__(self):
        self.is_multispeaker = len(self.speakers) > 1
        self.is_multilanguage = len(self.languages) > 1

    def prepare_input(
        self,
        text: str,
        lang: str | None = None,
        speaker: str | int | None = None,
        d_factor: float|None=None,
        p_factor: float|None=None,
        e_factor: float|None=None,
    ) -> InferenceInputs:
        sid = None
        lid = None
        phids, clean_text = self.tokenizer.tokenize(text=text, language=lang)
        
        phids = [phids]
        input_ids = []
        lengths = []
        for phid in phids:
            input_ids.append(phid)
            lengths.append(len(phid))
        sids = [sid] * len(input_ids) if sid is not None else None
        lids = [lid] * len(input_ids) if lid is not None else None
        return InferenceInputs.from_ids_and_lengths(
            ids=input_ids,
            lengths=lengths,
            clean_text=clean_text,
            sids=sids,
            lids=lids,
            d_factor=d_factor or self.inference_args["d_factor"],
            p_factor=p_factor or self.inference_args["p_factor"],
            e_factor=e_factor or self.inference_args["e_factor"],
        )

    def synthesise(self, inference_inputs: InferenceInputs) -> InferenceOutputs:
        inference_inputs = inference_inputs.as_numpy()
        synth_outs = self.synthesise_with_values(
            x=inference_inputs.x,
            x_lengths=inference_inputs.x_lengths,
            sids=inference_inputs.sids,
            lids=inference_inputs.lids,
            d_factor=inference_inputs.d_factor,
            p_factor=inference_inputs.p_factor,
            e_factor=inference_inputs.e_factor
        )
        return InferenceOutputs(
            wav=synth_outs["wav"],
            wav_lengths=synth_outs["wav_lengths"],
            latency=synth_outs["latency"],
            rtf=synth_outs["rtf"],
        )

    def synthesise_with_values(self, x, x_lengths, sids, lids, d_factor, p_factor, e_factor):
        inputs = dict(
            x=x,
            x_lengths=x_lengths,
            scales=np.array([d_factor, p_factor, e_factor], dtype=np.float32),
        )
        t0 = perf_counter()
        wav, wav_lengths, durations = self.session.run(None, inputs)
        t_infer = perf_counter() - t0
        t_audio = wav_lengths.sum() / self.sample_rate
        rtf = t_infer / t_audio
        latency = t_infer * 1000
        return dict(wav=wav, wav_lengths=wav_lengths, rtf=rtf, latency=latency)


