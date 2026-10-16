import csv
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent import main
from plot_results import main as plot_main, plot_results


class ExperimentTest(unittest.TestCase):
    def test_run_modes_metadata_and_plots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "new_output"
            for kind, training, evaluation in (
                ("random", 0, 3), ("random", 3, 0), ("q_learning", 3, 0),
                ("dqn", 0, 3), ("dqn", 3, 3),
            ):
                with self.subTest(agent=kind, training=training, evaluation=evaluation):
                    with redirect_stdout(StringIO()):
                        run_dir = main([
                            "--agent", kind, "--size", "3", "--agent_seed", "7",
                            "--game_seed", "11", "--eval_game_seed", "12",
                            "-tr", str(training), "-er", str(evaluation),
                            "--output_dir", str(root), "--plot",
                        ])
                    metadata = json.loads((run_dir / "run.json").read_text())
                    settings = metadata["settings"]
                    summary = json.loads((run_dir / "summary.json").read_text())
                    self.assertEqual((metadata["agent_config"]["seed"], settings["game_seed"], settings["evaluation_seed"]), (7, 11, 12))
                    if kind != "random":
                        self.assertEqual(metadata["agent_config"]["epsilon"], 1.0)
                    for phase, count in (("training", training), ("evaluation", evaluation)):
                        path = run_dir / f"{phase}.csv"
                        self.assertEqual(path.exists(), count > 0)
                        self.assertEqual(phase in summary, count > 0)
                        if count:
                            with path.open(newline="") as file:
                                rows = list(csv.DictReader(file))
                            self.assertEqual(len(rows), count)
                            self.assertEqual(summary[phase]["episodes"], count)
                            if phase == "evaluation" and kind != "random":
                                self.assertTrue(all(float(row["epsilon"]) == 0 for row in rows))
                    png = run_dir / "results.png"
                    self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    # Replotting consumes saved results, without invoking any agent.
                    with redirect_stdout(StringIO()):
                        self.assertEqual(plot_main([str(run_dir)]), png)

    def test_repeated_runs_do_not_overwrite_and_default_eval_seed_differs(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            args = ["--agent", "random", "-er", "1", "--size", "3",
                    "--agent_seed", "0", "--game_seed", "0", "--output_dir", directory]
            first = main(args)
            original = (first / "evaluation.csv").read_bytes()
            second = main(args)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "evaluation.csv").read_bytes(), original)
            self.assertEqual((second / "evaluation.csv").read_bytes(), original)
            settings = json.loads((second / "run.json").read_text())["settings"]
            self.assertEqual(settings["evaluation_seed"], 1)
            self.assertEqual(settings["training_runs"], 0)

    def test_empty_and_invalid_runs_write_nothing(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(directory) / "output"
            self.assertIsNone(main(["-tr", "0", "-er", "0", "--output_dir", str(root)]))
            for args in (["-tr", "-1"], ["--size", "2"], ["--game_seed", "1", "--eval_game_seed", "1"]):
                with self.assertRaises(SystemExit) as error:
                    main([*args, "--output_dir", str(root)])
                self.assertEqual(error.exception.code, 2)
            self.assertFalse(root.exists())

    def test_plot_rejects_empty_missing_and_nonfinite_data(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            for contents in ("", "steps\n1\n", "score\n", "score\noops\n", "score\nnan\n"):
                with self.subTest(contents=contents):
                    path.write_text(contents)
                    with self.assertRaises(ValueError):
                        plot_results(evaluation=path, output=Path(directory) / "bad.png")


if __name__ == "__main__":
    unittest.main()
