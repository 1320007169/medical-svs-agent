from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL_PATH = "/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx/DeepEyesV2/models/Qwen3-VL-8B-Instruct"


def test_sft_config_uses_local_model_and_step2_data():
    config = (ROOT / "configs" / "sft_qwen3_vl.yaml").read_text(encoding="utf-8")
    assert f'model_name_or_path: "{MODEL_PATH}"' in config
    assert "image_max_pixels: 262144" in config
    assert "dataset_dir: data_pipeline/step2_llamafactory" in config
    assert "template: qwen3_vl_nothink" in config


def test_cot_sft_config_uses_separate_data_and_output():
    config = (ROOT / "configs" / "sft_qwen3_vl_cot.yaml").read_text(encoding="utf-8")
    assert f'model_name_or_path: "{MODEL_PATH}"' in config
    assert "freeze_vision_tower: true" in config
    assert "freeze_multi_modal_projector: true" in config
    assert "dataset_dir: data_pipeline/step2_llamafactory_cot" in config
    assert "output_dir: outputs/sft/qwen3_vl_8b_medical_cot" in config


def test_combined_cot_sft_config_uses_new_data_and_output():
    config = (ROOT / "configs" / "sft_qwen3_vl_cot_1171.yaml").read_text(encoding="utf-8")
    assert f'model_name_or_path: "{MODEL_PATH}"' in config
    assert "freeze_vision_tower: true" in config
    assert "freeze_multi_modal_projector: true" in config
    assert "dataset_dir: data_pipeline/step2_llamafactory_cot_1171" in config
    assert "output_dir: outputs/sft/qwen3_vl_8b_medical_cot_1171" in config


def test_sft_launcher_checks_step2_data_and_uses_two_gpus():
    launcher = (ROOT / "scripts" / "run_sft.sh").read_text(encoding="utf-8")
    assert 'data_pipeline/step2_llamafactory/sft.jsonl' in launcher
    assert "SFT_DATA_FILE" in launcher
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"' in launcher
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-2}"' in launcher
    assert "FORCE_TORCHRUN=1" in launcher
    assert "SFT_CONDA_PREFIX=" in launcher
    assert 'SFT_PYTHON="$SFT_CONDA_PREFIX/bin/python"' in launcher
    assert 'exec "$SFT_PYTHON" -m torch.distributed.run' in launcher
    assert "GCC_PREFIX=" in launcher
    assert "TRITON_CACHE_DIR=" in launcher
    assert "PYTORCH_CUDA_ALLOC_CONF=" in launcher
    assert "TORIO_SOURCE=" in launcher
