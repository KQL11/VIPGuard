# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import (InferRequest, Model, ModelGroup, ModelMeta, PtEngine, RequestConfig, TemplateMeta,
                       get_model_tokenizer_with_flash_attn, register_model, register_template)
from typing import Any, Dict
from swift.llm.model.utils import AttnImpl, HfConfigFactory, ModelInfo, safe_snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.utils.versions import require_version
import transformers
from packaging import version
from peft import PeftModel
from swift.utils import get_dist_setting, get_logger, is_mp, is_unsloth_available, patch_getattr, use_torchacc
from swift.llm.model.patcher import (patch_automodel, patch_automodel_for_sequence_classification, patch_get_dynamic_module,
                      patch_mp_ddp)
from swift.llm.model.model_arch import MultiModelKeys, register_model_arch
import sys
sys.path.append('/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/Models')
from Qwen2_5_VL.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from Qwen2_5_VL.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from Qwen2_5_VL.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from functools import partial
logger = get_logger()
# /data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/ms-swift/swift/llm/model/utils.py


def _patch_awq_compat(model_info):
    if version.parse(transformers.__version__) < version.parse('4.50') or model_info.quant_method != 'awq':
        return

    try:
        # compat transformers>=4.50 (autoawq)
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        from transformers.integrations import get_keys_to_not_convert
        _process_model_before_weight_loading = AwqQuantizer._process_model_before_weight_loading

        def _new_process_model_before_weight_loading(self, model, *args, **kwargs):
            modules_to_not_convert = self.quantization_config.modules_to_not_convert
            if modules_to_not_convert is not None:
                self.quantization_config.modules_to_not_convert = list(
                    modules_to_not_convert) + get_keys_to_not_convert(model)
            return _process_model_before_weight_loading(self, model, *args, **kwargs)

        AwqQuantizer._process_model_before_weight_loading = _new_process_model_before_weight_loading
    except Exception:
        pass


def get_model_tokenizer_ours(model_dir: str,
                            model_info: ModelInfo,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'), kwargs.get('attn_impl_keys'))
    kwargs['model_config'] = model_config
    return get_model_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)



def get_model_tokenizer_from_local(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   *,
                                   tokenizer=None,
                                   model_config=None,
                                   automodel_class=AutoModel,
                                   **kwargs):
    """Load the model and tokenizer from the local model_dir."""
    if model_config is None:
        print(f'load model_config from {model_dir}')
        # model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model_config = Qwen2_5_VLVisionConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f'model_config: {model_config}')
    # fix prediction_step (internvl2, ovis, ...)
    if not hasattr(model_config, 'keys_to_ignore_at_inference'):
        model_config.keys_to_ignore_at_inference = []
    if 'past_key_values' not in model_config.keys_to_ignore_at_inference:
        model_config.keys_to_ignore_at_inference.append('past_key_values')

    torch_dtype = model_info.torch_dtype
    model_config.torch_dtype = torch_dtype
    HfConfigFactory.compat_zero3(model_config)
    rope_scaling = kwargs.get('rope_scaling')
    if rope_scaling:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if tokenizer is None:
        tokenizer = Qwen2_5_VLProcessor.from_pretrained(model_dir, trust_remote_code=True)
        # tokenizer = processor.tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    num_labels = model_info.num_labels or getattr(model_config, 'num_labels', None)
    if num_labels and model_info.task_type == 'seq_cls':
        model_info.num_labels = num_labels
        model_config.num_labels = num_labels

    model = None
    if load_model:
        _patch_awq_compat(model_info)
        logger.info(f'model_kwargs: {model_kwargs}')
        # fix seq_cls
        if model_info.task_type == 'seq_cls' and automodel_class is None:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
            except ValueError:
                model = None

        # automodel_class = automodel_class or AutoModelForCausalLM
        # NOTE:
        automodel_class = Qwen2_5_VLForConditionalGeneration
        model_meta = kwargs['model_meta']
        if model is None:
            if model_info.task_type == 'seq_cls' and not model_meta.is_reward:
                context = partial(patch_automodel_for_sequence_classification, model_meta=model_meta)
            else:
                context = partial(patch_automodel, automodel_class=automodel_class, model_info=model_info)
            with context():
                # model = automodel_class.from_pretrained(
                #     model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
                model = automodel_class.from_pretrained(
                    model_dir, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__

        if model_info.task_type == 'embedding' and automodel_class.__name__ != 'AutoModel':
            from swift.llm.model.patcher import patch_output_normalizer
            patch_output_normalizer(model, model_meta=model_meta)

    model_info.config = model_config if model is None else model.config
    if model:
        # fix seq classification task
        pad_token_id = model.config.pad_token_id or tokenizer.tokenizer.pad_token_id
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', pad_token_id)
    return model, tokenizer


# register_template(
#     TemplateMeta(
#         template_type='custom',
#         prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
#         prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
#         chat_sep=['\n']))

register_model(
    ModelMeta(
        model_type='kq_qwen2_5_vl',
        model_groups=[
            ModelGroup([Model('/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/Model_Stage1_Checkpoints/1', '/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/Model_Stage1_Checkpoints/1')]),
        ],
        template='kq_qwen2_5_vl',
        model_arch='kq_qwen2_5_vl',
        get_function=get_model_tokenizer_ours,
        ignore_patterns=['nemo']))
#
# if __name__ == '__main__':
#     infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
#     request_config = RequestConfig(max_tokens=512, temperature=0)
#     engine = PtEngine('AI-ModelScope/Nemotron-Mini-4B-Instruct')
#     response = engine.infer([infer_request], request_config)
#     swift_response = response[0].choices[0].message.content
#
#     engine.default_template.template_backend = 'jinja'
#     response = engine.infer([infer_request], request_config)
#     jinja_response = response[0].choices[0].message.content
#     assert swift_response == jinja_response, (f'swift_response: {swift_response}\njinja_response: {jinja_response}')
#     print(f'response: {swift_response}')
