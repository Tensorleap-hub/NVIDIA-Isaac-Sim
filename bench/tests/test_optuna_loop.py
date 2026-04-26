import numpy as np
from unittest.mock import patch, MagicMock
from convergence.optuna_loop import run_optuna_loop


def test_optuna_loop_smoke(tmp_path):
    real_embeddings = np.random.randn(50, 768).astype(np.float32)
    fake_embs = np.random.randn(4, 768).astype(np.float32)

    with patch("convergence.optuna_loop.Embedder") as MockEmbedder:
        mock_emb = MagicMock()
        mock_emb.embed.return_value = fake_embs
        MockEmbedder.return_value = mock_emb

        records = run_optuna_loop(
            real_embeddings=real_embeddings,
            run_dir=tmp_path,
            n_iterations=2,
            n_trials_per_iter=3,
            n_images=4,
            seed=42,
        )

    assert len(records) == 2
    assert all(r.best_objective >= 0.0 for r in records)
    assert (tmp_path / "metrics.csv").exists()


def test_optuna_loop_deterministic(tmp_path):
    real_embeddings = np.random.randn(50, 768).astype(np.float32)
    fake_embs = np.random.randn(4, 768).astype(np.float32)

    def run(run_id):
        with patch("convergence.optuna_loop.Embedder") as MockEmbedder:
            mock_emb = MagicMock()
            mock_emb.embed.return_value = fake_embs.copy()
            MockEmbedder.return_value = mock_emb
            return run_optuna_loop(
                real_embeddings=real_embeddings,
                run_dir=tmp_path / f"run{run_id}",
                n_iterations=2,
                n_trials_per_iter=2,
                n_images=4,
                seed=0,
            )

    r1 = run("A")
    r2 = run("B")
    assert abs(r1[0].best_objective - r2[0].best_objective) < 1e-6
