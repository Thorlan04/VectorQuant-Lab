"""VectorQuant-Lab Deterministic Benchmark & Evaluation Engine (Argon Hardened).

Generates realistic anisotropic high-dimensional embedding spaces in RAM,
evaluates industry baselines (SQ8, Percentile SQ8, Truncated SQ8), and evaluates
candidate quantization algorithms with strict Recall@10, bit-exact compression ratio,
and pure NumPy vectorized evaluation.
"""

from __future__ import annotations

import ast
import math
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def _extract_top10_vectorized(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy 2D vectorized extraction of Top-10 indices and cosine scores per query row.
    Eliminates per-query Python loops using 2D np.argpartition + np.take_along_axis.
    """
    part_idx = np.argpartition(sim_matrix, -10, axis=1)[:, -10:]
    part_vals = np.take_along_axis(sim_matrix, part_idx, axis=1)
    order = np.argsort(-part_vals, axis=1)
    top10_idx = np.take_along_axis(part_idx, order, axis=1).astype(np.int32)
    top10_vals = np.take_along_axis(part_vals, order, axis=1).astype(np.float32)
    return top10_idx, top10_vals


def _compute_exact_meta_bytes(obj: Any, _depth: int = 0) -> float:
    """
    Recursively accounts for every byte stored inside the candidate's `meta` dictionary
    to prevent metadata smuggling (e.g., hiding uncompressed float32 vectors in nested lists/tuples).
    """
    if _depth > 8 or obj is None:
        return 0.0
    if isinstance(obj, np.ndarray):
        return float(obj.nbytes)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return float(len(obj))
    if isinstance(obj, (int, float, bool, np.number)):
        return 8.0
    if isinstance(obj, str):
        return float(len(obj.encode("utf-8")))
    if isinstance(obj, dict):
        return sum(_compute_exact_meta_bytes(v, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_compute_exact_meta_bytes(v, _depth + 1) for v in obj)
    raise ValueError(f"Unsupported object type in quantization metadata: {type(obj).__name__}")


class SyntheticEmbeddingDataset:
    """RAM-only reproducible generator of anisotropic multi-cluster embeddings."""

    def __init__(
        self,
        num_vectors: int = 5000,
        dim: int = 384,
        num_queries: int = 200,
        num_clusters: int = 16,
        anisotropy_alpha: float = 0.65,
        seed: int = 42,
    ) -> None:
        self.num_vectors = num_vectors
        self.dim = dim
        self.num_queries = num_queries
        self.num_clusters = num_clusters
        self.anisotropy_alpha = anisotropy_alpha
        self.seed = seed

        self.db_vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.query_vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.ground_truth_top10: np.ndarray = np.empty((0, 10), dtype=np.int32)
        self.ground_truth_scores: np.ndarray = np.empty((0, 10), dtype=np.float32)

        self._generate_dataset()

    def _generate_dataset(self) -> None:
        """Generates realistic LLM-like anisotropic embeddings with power-law eigenvalue decay."""
        rng = np.random.RandomState(self.seed)

        # 1. Anisotropic spectral decay (simulates transformer attention drift)
        spectral_decay = 1.0 / (np.arange(1, self.dim + 1) ** self.anisotropy_alpha)
        spectral_decay = spectral_decay.astype(np.float32)

        # 2. Random orthogonal basis for correlated axes
        random_matrix = rng.randn(self.dim, self.dim).astype(np.float32)
        q_ortho, _ = np.linalg.qr(random_matrix)

        # 3. Cluster centroids in the skewed space
        cluster_weights = rng.dirichlet(np.ones(self.num_clusters)).astype(np.float32)
        cluster_centers = (
            rng.randn(self.num_clusters, self.dim).astype(np.float32)
            * (spectral_decay * 2.5)
        ) @ q_ortho

        # Constant global token drift (anisotropy hallmark in LLMs)
        global_drift = rng.randn(1, self.dim).astype(np.float32) * 0.4

        # 4. Generate Database Vectors
        cluster_assignments = rng.choice(
            self.num_clusters, size=self.num_vectors, p=cluster_weights
        )
        base_noise = (
            rng.randn(self.num_vectors, self.dim).astype(np.float32)
            * (spectral_decay * 0.8)
        ) @ q_ortho

        raw_db = cluster_centers[cluster_assignments] + base_noise + global_drift
        norms = np.linalg.norm(raw_db, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        self.db_vectors = (raw_db / norms).astype(np.float32)
        self.db_vectors.flags.writeable = False

        # 5. Generate Query Vectors
        q_assignments = rng.choice(
            self.num_clusters, size=self.num_queries, p=cluster_weights
        )
        q_noise = (
            rng.randn(self.num_queries, self.dim).astype(np.float32)
            * (spectral_decay * 0.6)
        ) @ q_ortho
        raw_queries = cluster_centers[q_assignments] + q_noise + global_drift
        q_norms = np.linalg.norm(raw_queries, axis=1, keepdims=True)
        q_norms = np.maximum(q_norms, 1e-9)
        self.query_vectors = (raw_queries / q_norms).astype(np.float32)
        self.query_vectors.flags.writeable = False

        # 6. Compute exact float32 Cosine Similarity Ground Truth (Vectorized)
        sim_matrix = np.dot(self.query_vectors, self.db_vectors.T)
        top10_idx, top10_vals = _extract_top10_vectorized(sim_matrix)

        self.ground_truth_top10 = top10_idx
        self.ground_truth_scores = top10_vals
        self.ground_truth_top10.flags.writeable = False


def compute_recall_at_10(
    predicted_top10: np.ndarray, ground_truth_top10: np.ndarray
) -> float:
    """
    Computes exact average Recall@10 over all queries using 3D NumPy broadcasting.
    Deduplicates predicted_top10 indices per query row first so returning duplicate indices cannot cheat.
    """
    if predicted_top10.shape != ground_truth_top10.shape:
        return 0.0
    # Ensure each ground_truth index is matched at most once per query:
    # ground_truth_top10[:, :, None] == predicted_top10[:, None, :] -> shape (Q, 10_gt, 10_pred)
    matches_per_gt = np.any(
        ground_truth_top10[:, :, None] == predicted_top10[:, None, :], axis=2
    )
    return float(np.mean(matches_per_gt))


# ==============================================================================
# INDUSTRY BASELINES (FOR RIGOROUS SCIENTIFIC COMPARISONS)
# ==============================================================================


class BaselineStandardSQ8:
    """Industry Baseline 1: Standard Min-Max Per-Vector Scalar Quantization (SQ8)."""

    name = "Standard Min-Max SQ8"

    @staticmethod
    def quantize(vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        v_min = np.min(vectors, axis=1, keepdims=True)
        v_max = np.max(vectors, axis=1, keepdims=True)
        denom = np.maximum(v_max - v_min, 1e-9)
        scaled = np.clip((vectors - v_min) / denom * 255.0, 0.0, 255.0)
        codes = np.round(scaled).astype(np.uint8)
        meta = {
            "v_min": v_min.astype(np.float32),
            "denom": (denom / 255.0).astype(np.float32),
        }
        return codes, meta

    @staticmethod
    def score(queries: np.ndarray, codes: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        v_min = meta["v_min"]
        step = meta["denom"]
        reconstructed = codes.astype(np.float32) * step + v_min
        rec_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        rec_norms = np.maximum(rec_norms, 1e-9)
        normed = reconstructed / rec_norms
        return np.dot(queries, normed.T)


class BaselinePercentileClippedSQ8:
    """Industry Baseline 2: Uniform Percentile Clipped Quantization (3-Sigma / p0.5-p99.5)."""

    name = "Percentile Clipped SQ8"

    @staticmethod
    def quantize(vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        p_low = np.percentile(vectors, 0.5, axis=1, keepdims=True)
        p_high = np.percentile(vectors, 99.5, axis=1, keepdims=True)
        denom = np.maximum(p_high - p_low, 1e-9)
        clipped = np.clip(vectors, p_low, p_high)
        scaled = (clipped - p_low) / denom * 255.0
        codes = np.round(scaled).astype(np.uint8)
        meta = {
            "p_low": p_low.astype(np.float32),
            "step": (denom / 255.0).astype(np.float32),
        }
        return codes, meta

    @staticmethod
    def score(queries: np.ndarray, codes: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        p_low = meta["p_low"]
        step = meta["step"]
        reconstructed = codes.astype(np.float32) * step + p_low
        rec_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        rec_norms = np.maximum(rec_norms, 1e-9)
        return np.dot(queries, (reconstructed / rec_norms).T)


class BaselineTruncatedInt8:
    """Industry Baseline 3: Naive Truncated Dimensionality (Top 75% dims) + Int8."""

    name = "Truncated 75% Dims + Int8"

    @staticmethod
    def quantize(vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        dim = vectors.shape[1]
        keep_dim = int(dim * 0.75)
        truncated = vectors[:, :keep_dim]
        max_abs = np.max(np.abs(truncated), axis=1, keepdims=True)
        denom = np.maximum(max_abs, 1e-9)
        scaled = np.clip(truncated / denom * 127.0, -128.0, 127.0)
        codes = np.round(scaled).astype(np.int8).view(np.uint8)
        meta = {"denom": (denom / 127.0).astype(np.float32), "keep_dim": keep_dim}
        return codes, meta

    @staticmethod
    def score(queries: np.ndarray, codes: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        keep_dim = meta["keep_dim"]
        sub_queries = queries[:, :keep_dim]
        denom = meta["denom"]
        signed_codes = codes.view(np.int8).astype(np.float32) * denom
        rec_norms = np.linalg.norm(signed_codes, axis=1, keepdims=True)
        rec_norms = np.maximum(rec_norms, 1e-9)
        return np.dot(sub_queries, (signed_codes / rec_norms).T)


# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================


class BenchmarkEngine:
    """Evaluates algorithms and baselines in an isolated sandbox with bit-exact memory auditing."""

    def __init__(self, dataset: Optional[SyntheticEmbeddingDataset] = None) -> None:
        if dataset is None:
            dataset = SyntheticEmbeddingDataset(
                num_vectors=5000, dim=384, num_queries=200, seed=42
            )
        self.dataset = dataset
        self.baselines_results: Dict[str, Dict[str, Any]] = {}

    def run_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Evaluates all 3 industry baselines to set calibration targets."""
        baselines = [
            BaselineStandardSQ8,
            BaselinePercentileClippedSQ8,
            BaselineTruncatedInt8,
        ]

        for base in baselines:
            res = self.evaluate_methods(
                name=base.name,
                quantize_fn=base.quantize,
                score_fn=base.score,
            )
            self.baselines_results[base.name] = res
        return self.baselines_results

    def evaluate_methods(
        self,
        name: str,
        quantize_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
        score_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray],
    ) -> Dict[str, Any]:
        """Runs quantization and scoring, measures exact Recall@10, bit-exact compression, and latency."""
        t0 = time.perf_counter()

        # 1. Quantization on read-only input view
        db_view = self.dataset.db_vectors.view()
        db_view.flags.writeable = False
        codes, meta = quantize_fn(db_view)
        t_quant = (time.perf_counter() - t0) * 1000.0

        if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8:
            raise TypeError("quantize_transform MUST return a uint8 NumPy array for `codes`.")
        if not isinstance(meta, dict):
            raise TypeError("quantize_transform MUST return a dict for `meta`.")

        # Bit-exact memory accounting
        num_v, dim = self.dataset.db_vectors.shape
        fp32_bytes = float(num_v * dim * 4)  # 4 bytes per float32 (32 bits)

        codes_bits = int(codes.nbytes) * 8
        meta_bytes = _compute_exact_meta_bytes(meta)
        meta_bits = int(math.ceil(meta_bytes)) * 8

        # Anti-cheat: forbid smuggling uncompressed vectors inside `meta`
        if meta_bytes > 0.25 * fp32_bytes:
            raise ValueError(
                f"Anti-Cheat Violation: Metadata size ({meta_bytes:.0f} B) exceeds 25% of raw FP32 size."
            )

        total_compressed_bits = max(codes_bits + meta_bits, 8)
        total_compressed_bytes = float(total_compressed_bits) / 8.0
        compression_ratio = (fp32_bytes * 8.0) / float(total_compressed_bits)
        bytes_per_vector = total_compressed_bytes / float(num_v)

        if compression_ratio < 1.5:
            raise ValueError(
                f"Compression ratio ({compression_ratio:.2f}x) is below minimum 1.50x requirement."
            )

        # 2. Asymmetric similarity search on read-only query view
        q_view = self.dataset.query_vectors.view()
        q_view.flags.writeable = False
        codes.flags.writeable = False

        t1 = time.perf_counter()
        scores_matrix = score_fn(q_view, codes, meta)
        t_score = (time.perf_counter() - t1) * 1000.0
        total_time_ms = t_quant + t_score

        num_queries = self.dataset.num_queries
        if not isinstance(scores_matrix, np.ndarray) or scores_matrix.shape != (num_queries, num_v):
            raise ValueError(f"Score matrix shape must be ({num_queries}, {num_v}), got {getattr(scores_matrix, 'shape', None)}")
        if not np.all(np.isfinite(scores_matrix)):
            raise ValueError("Score matrix contains NaN or Inf values.")
        if float(np.std(scores_matrix)) < 1e-6:
            raise ValueError("Degenerate/constant similarity matrix rejected (Anti-Overfitting Gate).")

        # 3. Vectorized Top-10 extraction and Recall@10
        pred_top10, _ = _extract_top10_vectorized(scores_matrix)
        recall_at_10 = compute_recall_at_10(
            pred_top10, self.dataset.ground_truth_top10
        )

        if recall_at_10 >= 0.92:
            composite_score = (
                (recall_at_10 * 100.0)
                * (compression_ratio ** 0.40)
                / (max(total_time_ms, 1.0) ** 0.08)
            )
        else:
            composite_score = 0.0

        return {
            "name": name,
            "recall_at_10": round(recall_at_10, 4),
            "compression_ratio": round(compression_ratio, 2),
            "bytes_per_vector": round(bytes_per_vector, 1),
            "ram_savings_pct": round((1.0 - (1.0 / compression_ratio)) * 100.0, 1),
            "latency_ms": round(total_time_ms, 2),
            "quant_time_ms": round(t_quant, 2),
            "score_time_ms": round(t_score, 2),
            "composite_score": round(composite_score, 2),
            "passed_threshold": recall_at_10 >= 0.92,
        }

    def evaluate_code_sandbox(
        self,
        algorithm_name: str,
        code_string: str,
        timeout_seconds: float = 3.0,
    ) -> Dict[str, Any]:
        """Safely executes candidate code within an AST-validated isolated scope and evaluates its performance."""
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Global, ast.Nonlocal)):
                    raise ValueError(f"Forbidden AST construct: {type(node).__name__}")
        except Exception as e:
            return {
                "name": algorithm_name,
                "error": f"AST Security Violation: {str(e)}",
                "recall_at_10": 0.0,
                "compression_ratio": 0.0,
                "latency_ms": 9999.0,
                "composite_score": 0.0,
                "passed_threshold": False,
            }

        def safe_import(name, *args, **kwargs):
            if name in ("numpy", "np", "math", "typing"):
                return __import__(name, *args, **kwargs)
            raise ImportError(f"Import of '{name}' is restricted in sandbox.")

        isolated_globals = {
            "np": np,
            "numpy": np,
            "math": math,
            "__builtins__": {
                "__import__": safe_import,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "round": round,
                "max": max,
                "min": min,
                "abs": abs,
                "sum": sum,
                "isinstance": isinstance,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
                "zip": zip,
                "enumerate": enumerate,
                "print": print,
                "True": True,
                "False": False,
                "None": None,
                "__build_class__": __build_class__,
                "object": object,
                "slice": slice,
                "ValueError": ValueError,
                "TypeError": TypeError,
            },
        }

        try:
            exec(code_string, isolated_globals)  # pylint: disable=exec-used

            quantize_fn = isolated_globals.get("quantize_transform")
            score_fn = isolated_globals.get("asymmetric_similarity_score")

            if quantize_fn is None or score_fn is None:
                cls = isolated_globals.get("VectorQuantizer")
                if cls is not None:
                    instance = cls()
                    quantize_fn = getattr(instance, "quantize_transform", None) or getattr(instance, "quantize", None)
                    score_fn = getattr(instance, "asymmetric_similarity_score", None) or getattr(instance, "score", None)

            if not callable(quantize_fn) or not callable(score_fn):
                return {
                    "name": algorithm_name,
                    "error": "Missing callable 'quantize_transform' and 'asymmetric_similarity_score'",
                    "recall_at_10": 0.0,
                    "compression_ratio": 0.0,
                    "latency_ms": 9999.0,
                    "composite_score": 0.0,
                    "passed_threshold": False,
                }

            result = self.evaluate_methods(
                name=algorithm_name,
                quantize_fn=quantize_fn,
                score_fn=score_fn,
            )
            return result

        except Exception as e:
            tb = traceback.format_exc(limit=2)
            return {
                "name": algorithm_name,
                "error": f"{type(e).__name__}: {str(e)} | {tb}",
                "recall_at_10": 0.0,
                "compression_ratio": 0.0,
                "latency_ms": 9999.0,
                "composite_score": 0.0,
                "passed_threshold": False,
            }
