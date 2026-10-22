"""Small structural checks; no training-set download or full-size training."""
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel

from detector.embedding import MuRIL
from scripts.full_finetune import grouped_folds, inner_split
from tests.test_pipeline import detector


def test_outer_and_inner_group_boundaries():
    groups = np.repeat(np.arange(100), 3)
    data = {'text': np.array(['example'] * len(groups)), 'group': groups,
            'y': np.tile([0, 1, 2], 100)}
    seen = []
    for fold, (fit, holdout) in enumerate(grouped_folds(data)):
        train, stop = inner_split(data, fit, fold)
        seen.extend(holdout)
        partitions = [set(groups[x]) for x in (train, stop, holdout)]
        assert not partitions[0] & partitions[1]
        assert not partitions[0] & partitions[2]
        assert not partitions[1] & partitions[2]
        assert set(train) | set(stop) == set(fit)
    assert sorted(seen) == list(range(len(groups)))


def test_full_checkpoint_serves_updated_embeddings_and_layers(tmp_path):
    torch.manual_seed(1)
    config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=32)
    encoder = MuRIL.__new__(MuRIL)
    encoder.model = BertModel(config).eval()
    encoder.torch = torch
    encoder.task_layers, encoder.task_models = {}, {}
    def tokenize(texts, **kwargs):
        return dict(input_ids=torch.tensor([[1, 2, 3]] * len(texts)),
                    attention_mask=torch.ones((len(texts), 3), dtype=torch.long))
    encoder.tokenizer = tokenize
    updated = BertModel(config, add_pooling_layer=False).eval()
    with torch.no_grad():
        expected = updated(**tokenize(['x'])).last_hidden_state.mean(1).numpy()
    path = tmp_path / 'full.safetensors'
    save_file(updated.state_dict(), str(path))
    encoder.configure_full_task('test', path)
    np.testing.assert_allclose(encoder.encode(['x'], task='test'), expected, atol=1e-6)
    assert not np.allclose(encoder.encode(['x']), expected)


def test_full_encoder_attribution_gets_its_own_embeddings():
    service = detector([.1, .9])
    service.artifacts['task2_attribution']['metadata']['encoder'] = {'fine_tuned_full': 'family.safetensors'}
    calls = []
    def encode(texts, task=None):
        calls.append(task)
        return np.zeros((len(texts), 768))
    service.encoder.encode = encode
    service.predict('this passage has enough words for a family prediction', 'en')
    assert calls == ['task1_binary', 'task2_attribution']


def test_epoch_checkpoint_resumes_after_interruption(tmp_path, monkeypatch):
    import pytest
    import scripts.full_finetune as experiment
    from transformers import BertTokenizer
    torch.set_num_threads(2)
    monkeypatch.setattr(experiment, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(experiment, 'CACHE', tmp_path)
    config = BertConfig(vocab_size=8, hidden_size=768, num_hidden_layers=1,
                        num_attention_heads=12, intermediate_size=32)
    monkeypatch.setattr(experiment, 'new_model', lambda: BertModel(config, add_pooling_layer=False))
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\none\ntwo\nthree\n')
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    rng = np.random.default_rng(1)
    data = dict(text=np.array(['one two three'] * 12), emb=rng.normal(size=(12, 768)),
                y=np.tile([0, 1, 2], 4).astype(np.int32), classes=np.array(['a', 'b', 'c']))
    real_checkpoint = experiment.checkpoint
    def interrupt_after_checkpoint(*args, **kwargs):
        real_checkpoint(*args, **kwargs)
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(experiment, 'checkpoint', interrupt_after_checkpoint)
    folder = tmp_path / 'checkpoint'
    with pytest.raises(RuntimeError, match='simulated interruption'):
        experiment.train_encoder(tokenizer, data, np.arange(9), np.arange(9, 12), 2e-6,
                                 folder, dict(task='tiny', fold=0), epochs=2)
    monkeypatch.setattr(experiment, 'checkpoint', real_checkpoint)
    _, result = experiment.train_encoder(tokenizer, data, np.arange(9), np.arange(9, 12), 2e-6,
                                         folder, dict(task='tiny', fold=0), epochs=2)
    assert [row['epoch'] for row in result['history']] == [1, 2]
    assert all(value > 0 for value in result['first_batch_gradient_norms'].values())
