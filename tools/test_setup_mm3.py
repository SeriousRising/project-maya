"""tools/test_setup_mm3.py - the setup's MiniMax-M3.1-Flash (maya.py): where its EXL3 quant is offered, the context it
recommends by the PC's memory (and its warnings), its config - the model card's sampling, the thinking level, pictures
on - and what a setup again keeps.  No GPU, no network.

    python -m unittest tools.test_setup_mm3 -v
"""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402


class Failed(Exception):
    pass


def args(**kw):
    base = dict(model=maya.MM3, model_dir=None, gguf_dir=None, draft=None, backend="cuda", yes=True, port=None,
                host=None, api_key=None, env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None)
    base.update(kw)
    return SimpleNamespace(**base)


def fail(msg, hint=None):
    raise Failed(msg)


class Gate(unittest.TestCase):
    """As DeepSeek's EXL3: NVIDIA only (AMD refused with a clear message); Windows and V100 once the engine runs there
    (MM3_ALL_PLATFORMS, one switch)."""

    def check(self, backend="cuda", win=False, everyone=False, gpus=None):
        with patch.object(maya, "fail", fail), patch.object(maya, "WIN", win), \
                patch.object(maya, "MM3_ALL_PLATFORMS", everyone), patch.object(maya, "step", lambda *a: None), \
                patch.object(maya.S, "gpus", return_value=gpus or []), patch.object(maya, "say", lambda *a: None), \
                patch.object(maya, "warn", lambda *a: None), patch.object(maya.S, "is_wsl", return_value=False):
            maya.check_pc(args(backend=backend), "minimax")

    def test_amd_refused(self):
        for everyone in (False, True):
            with self.assertRaisesRegex(Failed, "NVIDIA GPUs only"):
                self.check(backend="hip", everyone=everyone)

    def test_windows_waits_for_the_switch(self):
        with self.assertRaisesRegex(Failed, "Linux only for now"):
            self.check(win=True)
        with self.assertRaisesRegex(Failed, "no NVIDIA GPU"):       # (past the gate: the GPU check)
            self.check(win=True, everyone=True)

    def test_min_cc(self):
        with patch.object(maya, "warn", lambda *a: None):
            with patch.object(maya, "MM3_ALL_PLATFORMS", False), patch.dict(maya.os.environ, {}, clear=False):
                maya.os.environ.pop(maya.MM3_TEST_CC_ENV, None)
                self.assertEqual(maya.mm3_min_cc(), 80)
                maya.os.environ[maya.MM3_TEST_CC_ENV] = "70"
                self.assertEqual(maya.mm3_min_cc(), 70)               # (testing only, opt-in)
                maya.os.environ.pop(maya.MM3_TEST_CC_ENV)
            with patch.object(maya, "MM3_ALL_PLATFORMS", True):
                self.assertEqual(maya.mm3_min_cc(), maya.DS4_MIN_CC)   # (DeepSeek's EXL3 reach: V100 on)
        self.assertEqual(maya.MM3_MIN_CC, maya.DS4_MIN_CC)

    def test_exe_by_platform(self):
        self.assertEqual(maya.exe_for_target("maya-mm3"), maya.MM3_EXE)
        maya.select_build_backend("cuda")
        self.assertEqual(maya.MM3_EXE.name, "maya-mm3.exe" if maya.WIN else "maya-mm3")


class Contexts(unittest.TestCase):
    def test_up_to_the_models_own(self):
        self.assertEqual(maya.MM3_CONTEXTS, [32768, 65536, 131072, 262144, 524288, 1048576])

    def test_recommended_by_memory(self):
        """The largest tested size whose KV cache fits beside the model in VRAM + RAM; the smallest when it does not."""
        bf16 = maya.MM3_KV_BYTES["bf16"]
        self.assertAlmostEqual(maya.mm3_kv_gb(1048576, "bf16"), 1048576 * bf16 / 2**30)
        self.assertEqual(maya.mm3_recommended_context(48, 512, "bf16"), 262144)    # (never past the tested 256K)
        self.assertEqual(maya.mm3_recommended_context(48, 256, "bf16"), 262144)    # (54 GB spare: 30 GB at 256K)
        self.assertEqual(maya.mm3_recommended_context(48, 220, "bf16"), 131072)    # (18 GB spare: 15 GB at 128K)
        self.assertEqual(maya.mm3_recommended_context(24, 128, "bf16"), 32768)     # (below the model: the smallest)
        self.assertEqual(maya.mm3_recommended_context(48, 0, "bf16"), maya.MM3_DEFAULT_CONTEXT)   # (RAM unknown)
        self.assertGreater(maya.mm3_recommended_context(48, 214, "int8"), maya.mm3_recommended_context(48, 214, "bf16"))

    def test_kv_format_from_the_config(self):
        self.assertEqual(maya.mm3_kv_format(["--kv-cache", "int8"]), "int8")
        self.assertEqual(maya.mm3_kv_format([]), maya.MM3_KV_FORMAT)

    def test_warnings(self):
        said = []
        with patch.object(maya, "warn", said.append), patch.object(maya, "ok", lambda *a: None):
            maya.mm3_context_notes(262144, 48, 256, "bf16")
            self.assertEqual(said, [])
            maya.mm3_context_notes(1048576, 48, 256, "bf16")
        self.assertEqual(len(said), 2)
        self.assertIn("not been run end to end", said[0])
        self.assertIn("256K fits this PC", said[1])


class Config(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pack = self.root / "MM" / "maya-mm3"
        self.pack.mkdir(parents=True)
        (self.pack / "mm3.txt").write_text("hidden_size 6144\nmax_position_embeddings 1048576\n"
                                           "vision.hidden_size 1024\ndspark.num_hidden_layers 5\n")
        self.patches = [patch.object(maya, "ROOT", self.root), patch.object(maya, "say", lambda *a: None),
                        patch.object(maya, "ok", lambda *a: None), patch.object(maya, "step", lambda *a: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def write(self, a, old=None, gpus=3):
        if old is not None:
            (self.root / "maya-mm3.json").write_text(json.dumps(old))
        pc = {"gpus": [{"index": i} for i in range(gpus)]}
        p = maya.write_config_mm3(a, pc, {"lib_dirs": []}, self.pack, 131072, Path("/models"), self.pack.parent)
        return json.loads(p.read_text())

    def test_fresh(self):
        c = self.write(args())
        a = c["args"]
        self.assertEqual((c["sampling"], c["reasoning_effort"]), ({"temperature": 1.0, "top_p": 0.95}, "auto"))
        self.assertEqual(a[a.index("--vision") + 1], "on")                # (pictures on by default)
        self.assertEqual(a[a.index("--dspark") + 1], "off")             # (until this PC's speed calibration)
        self.assertEqual(a[a.index("--kv-streaming") + 1], "on")
        self.assertEqual(a[a.index("--max-context") + 1], "131072")
        self.assertEqual(c["model_name"], "minimax-m3.1-flash")
        self.assertEqual(maya.config_family(c), "minimax")

    def test_no_vision(self):
        a = self.write(args(no_vision=True))["args"]
        self.assertEqual(a[a.index("--vision") + 1], "off")

    def test_again_keeps_what_it_does_not_ask(self):
        old = {"args": ["--pack", "/old", "--max-context", "262144", "--kv-cache", "int8", "--vision", "off"],
               "reasoning_effort": "high", "sampling": {"temperature": 0.7}, "host": "0.0.0.0"}
        c = self.write(args(), old)
        a = c["args"]
        self.assertEqual(a[a.index("--kv-cache") + 1], "int8")
        self.assertEqual(a[a.index("--vision") + 1], "off")
        self.assertEqual((c["reasoning_effort"], c["sampling"], c["host"]), ("high", {"temperature": 0.7}, "0.0.0.0"))

    def test_dspark_measured_as_deepseeks(self):
        """The DSpark drafter: off in the config until this PC's speed calibration (DeepSeek's tool) says it pays."""
        import calibrate_ds4 as CDS
        c = self.write(args())
        self.assertFalse(CDS.has_drafter(c))                               # (no dspark.* tensors indexed yet)
        (self.pack / "tensors.txt").write_text("file 0 a.safetensors\nt dspark.fc.weight BF16 0 0 8 2 2\n")
        self.assertTrue(CDS.has_drafter(c))
        self.assertTrue(maya.needs_speed_calibration(c))
        a = CDS.with_dspark(c, "none")["args"]
        self.assertEqual(a[a.index("--dspark") + 1], "off")                # (maya-mm3's word for none)
        self.assertEqual(CDS.profile_copy(c, self.root)["args"], c["args"])  # (it only reads its --profile)

    def test_template_in_the_pack_tools(self):
        """A change to Maya's MiniMax template rebuilds the pack (its tokenizer folder carries the template)."""
        self.assertIn("tools/mm3/chat_template.jinja", maya.MM3_PACK_TOOLS)
        self.assertTrue((Path(maya.__file__).parent / "tools" / "mm3" / "chat_template.jinja").exists())


if __name__ == "__main__":
    unittest.main()
