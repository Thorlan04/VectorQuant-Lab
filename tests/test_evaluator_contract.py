"""
VectorQuant-Lab — Formal Evaluator Contract & Anti-Cheat Regression Suite
Covers invalid subcases (runtime exceptions, NaN/Inf similarity matrices,
constant/degenerate outputs, metadata smuggling >25%, AST violations)
plus valid finite-score controls.
"""

import unittest
from src.benchmark_engine import BenchmarkEngine, SyntheticEmbeddingDataset


class TestEvaluatorContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = SyntheticEmbeddingDataset(num_vectors=200, dim=64, num_queries=20, seed=42)
        cls.engine = BenchmarkEngine(dataset=dataset)

    def test_01_runtime_exceptions_and_ast_violations_rejected(self):
        subcases = [
            """
def quantize_transform(vectors_fp32, calibration_stats=None):
    return 1 / 0, {}
def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    return query_fp32 @ query_fp32.T
""",
            """
def quantize_transform(vectors_fp32, calibration_stats=None):
    global _hack
    return vectors_fp32.astype(np.uint8), {}
def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    return np.zeros((query_fp32.shape[0], quantized_codes.shape[0]))
""",
            """
def quantize_transform(vectors_fp32, calibration_stats=None):
    vectors_fp32[0, 0] = 0.0
    return np.zeros_like(vectors_fp32, dtype=np.uint8), {}
def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    return np.ones((query_fp32.shape[0], quantized_codes.shape[0]))
""",
        ]
        for idx, code in enumerate(subcases, 1):
            with self.subTest(subcase=idx):
                res = self.engine.evaluate_code_sandbox(f"Invalid_{idx}", code)
                self.assertFalse(res["passed_threshold"])
                self.assertEqual(res["composite_score"], 0.0)

    def test_02_non_finite_degenerate_and_metadata_smuggling_rejected(self):
        subcases = [
            # NaN score matrix
            """
def quantize_transform(vectors_fp32, calibration_stats=None):
    return np.zeros_like(vectors_fp32, dtype=np.uint8), {}
def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    mat = np.zeros((query_fp32.shape[0], quantized_codes.shape[0]), dtype=np.float32)
    mat[:] = np.nan
    return mat
""",
            # Constant degenerate matrix
            """
def quantize_transform(vectors_fp32, calibration_stats=None):
    return np.zeros_like(vectors_fp32, dtype=np.uint8), {}
def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    return np.ones((query_fp32.shape[0], quantized_codes.shape[0]), dtype=np.float32)
""",
            # Metadata smuggling (>25% of raw FP32 bytes)
            """
def quantize_transform(vectors_fp32, calibration_stats=None):
    return np.zeros_like(vectors_fp32, dtype=np.uint8), {"raw_copy": vectors_fp32.copy()}
def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    return query_fp32 @ meta_params["raw_copy"].T
""",
        ]
        for idx, code in enumerate(subcases, 1):
            with self.subTest(subcase=idx):
                res = self.engine.evaluate_code_sandbox(f"NonFinite_{idx}", code)
                self.assertFalse(res["passed_threshold"])
                self.assertEqual(res["composite_score"], 0.0)

    def test_03_valid_finite_score_control_passes(self):
        valid_code = """
def quantize_transform(vectors_fp32, calibration_stats=None):
    norms = np.maximum(np.linalg.norm(vectors_fp32, axis=1, keepdims=True), 1e-9)
    v = vectors_fp32 / norms
    scale = np.maximum(np.max(np.abs(v), axis=1, keepdims=True), 1e-9) / 127.0
    codes = np.clip(np.round(v / scale), -127, 127).astype(np.int8).view(np.uint8)
    return codes, {"scale": scale.astype(np.float16)}

def asymmetric_similarity_score(query_fp32, quantized_codes, meta_params):
    scale = meta_params["scale"].astype(np.float32)
    rec = quantized_codes.view(np.int8).astype(np.float32) * scale
    rec /= np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-9)
    q = query_fp32 / np.maximum(np.linalg.norm(query_fp32, axis=1, keepdims=True), 1e-9)
    return np.dot(q, rec.T)
"""
        res = self.engine.evaluate_code_sandbox("ValidControl", valid_code)
        self.assertTrue(res["passed_threshold"], f"Valid control failed: {res.get('error')}")
        self.assertGreater(res["composite_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
