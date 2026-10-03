"""Numpy-backed test doubles verify HF adapter plumbing, not real model quality."""

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from search_agent.config import BGE_ID, QWEN_ID, ModelConfig
from search_agent.models import BGEReranker, ModelError, QwenEmbedding


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data.tolist()

    def view(self, *shape):
        return Tensor(self.data.reshape(shape))


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def hf(monkeypatch):
    tokenizer = Mock(return_value=Inputs(input_ids=Tensor([[1, 2]])))
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config = SimpleNamespace(_commit_hash="immutable-test-sha")
    embedding_factory = SimpleNamespace(from_pretrained=Mock(return_value=model))
    reranker_factory = SimpleNamespace(from_pretrained=Mock(return_value=model))
    tokenizer_factory = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            inference_mode=contextlib.nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=embedding_factory,
            AutoModelForSequenceClassification=reranker_factory,
            AutoTokenizer=tokenizer_factory,
        ),
    )
    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        embedding=embedding_factory,
        reranker=reranker_factory,
        tokenizer_factory=tokenizer_factory,
    )


def test_qwen_load_pool_dimensions_normalization_and_revision(hf):
    values = np.zeros((1, 2, 2560))
    values[:, 0, 0] = 99  # Must pool the final real token, not the first.
    values[:, -1, :2] = [3, 4]
    hf.model.return_value = SimpleNamespace(last_hidden_state=Tensor(values))
    adapter = QwenEmbedding(ModelConfig(revision="pinned-sha"), "/tmp/test-cache")
    result = adapter.documents(["한국어 문서"])
    assert result.shape == (1, 2560)
    assert np.allclose(result[0, :2], [0.6, 0.8])
    assert np.linalg.norm(result[0]) == pytest.approx(1)
    assert adapter.identity["resolved_revision"] == "immutable-test-sha"
    args, kwargs = hf.embedding.from_pretrained.call_args
    assert args == (QWEN_ID,)
    assert kwargs["revision"] == "pinned-sha" and kwargs["trust_remote_code"] is False
    assert hf.tokenizer_factory.from_pretrained.call_args.kwargs["padding_side"] == "left"
    assert hf.tokenizer.call_args.args[0] == ["한국어 문서"]
    adapter.query("공룡")
    assert hf.tokenizer.call_args.args[0][0].startswith("Instruct:")
    assert hf.embedding.from_pretrained.call_count == 1


def test_qwen_truncation_before_normalization(hf):
    values = np.ones((1, 1, 2560))
    hf.model.return_value = SimpleNamespace(last_hidden_state=Tensor(values))
    adapter = QwenEmbedding(ModelConfig(), "/tmp/test-cache", dimensions=32)
    result = adapter.documents(["한국어"])
    assert result.shape == (1, 32)
    assert np.linalg.norm(result[0]) == pytest.approx(1)


def test_bge_query_passage_pairs_raw_logits_single_and_batches(hf):
    hf.model.return_value = SimpleNamespace(logits=Tensor([[2.5]]))
    adapter = BGEReranker(ModelConfig(), "/tmp/test-cache")
    assert adapter.score("공룡", ["문서 하나", "문서 둘"]) == [2.5, 2.5]
    assert hf.tokenizer.call_args_list[0].args[0] == [["공룡", "문서 하나"]]
    assert hf.tokenizer.call_args_list[1].args[0] == [["공룡", "문서 둘"]]
    assert hf.reranker.from_pretrained.call_args.args == (BGE_ID,)
    assert hf.tokenizer.call_args.kwargs["max_length"] == 512
    assert adapter.score("공룡", ["한 개"]) == [2.5]
    assert adapter.score("공룡", []) == []


@pytest.mark.parametrize(
    "adapter_type,stage", [(QwenEmbedding, "embedding"), (BGEReranker, "reranker")]
)
def test_load_failure_stays_independent(hf, adapter_type, stage):
    hf.tokenizer_factory.from_pretrained.side_effect = OSError("offline")
    adapter = adapter_type(ModelConfig(), "/tmp/test-cache")
    with pytest.raises(ModelError) as failure:
        adapter._load()
    assert failure.value.stage == stage
    assert adapter.model is None
    assert not hf.embedding.from_pretrained.called
    assert not hf.reranker.from_pretrained.called
