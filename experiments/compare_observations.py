"""Run matched Q-learning seeds with only the observation width changed."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New experiment folder')
    parser.add_argument('--training-runs', type=int, default=5000)
    parser.add_argument('--evaluation-runs', type=int, default=300)
    parser.add_argument('--resume-from', type=Path, help='Previous experiment to extend to --training-runs total games')
    args = parser.parse_args()
    if args.training_runs < 1 or args.evaluation_runs < 1:
        parser.error('Episode counts must be positive')
    previous = None
    if args.resume_from:
        previous = json.loads((args.resume_from / 'experiment.json').read_text(encoding='utf-8'))['results']
        if any(row['training_games'] >= args.training_runs for row in previous):
            parser.error('The target must exceed the previous training budget')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    commands, summaries = [], []
    for seed in (7, 17, 27):
        for mode in ('standard', 'diagonal'):
            name = f'{mode}-{args.training_runs}-seed-{seed}' if previous else f'{mode}-seed-{seed}'
            run = root / name
            completed = 0
            start = ['--agent', 'q_learning', '--size', '6', '--agent_seed', str(seed), '--game_seed', str(seed + 1)]
            if previous:
                prior = next(row for row in previous if row['mode'] == mode and row['agent_seed'] == seed)
                completed = prior['training_games']
                saved = args.resume_from.resolve() / prior['run'] / f'checkpoints/episode_{completed:08d}.json'
                start = ['--load-checkpoint', str(saved), '--resume-training']
            elif mode == 'diagonal':
                start.append('--diagonal-obstacles')
            command = [sys.executable, str(source / 'agent.py'), *start, '--eval-seed', '101',
                       '--eval-agent-seed', '0', '-tr', str(args.training_runs - completed), '-er', str(args.evaluation_runs),
                       '--record', 'detailed', '--record-every', '250' if previous else '50', '--record-episodes', '0',
                       '--checkpoint-every', '1000', '--output-dir', str(run)]
            print(f'Running {name}...', flush=True)
            with (root / f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            commands.append({'run': name, 'command': command})
            with (run / 'evaluation.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            scores = [int(row['score']) for row in rows]
            checkpoint = json.loads((run / f'checkpoints/episode_{args.training_runs:08d}.json').read_text())
            summary = dict(run=name, mode=mode, agent_seed=seed, game_seed=seed + 1,
                           evaluation_seed=101, evaluation_agent_seed=0, training_games=args.training_runs,
                           resumed_from_games=completed,
                           evaluation_games=len(rows), mean_score=statistics.mean(scores),
                           median_score=statistics.median(scores), best_score=max(scores),
                           mean_steps=statistics.mean(int(row['steps']) for row in rows),
                           q_table_states=len(checkpoint['q_table']),
                           starvation_games=sum(row['termination_reason'] == 'starvation_limit' for row in rows))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    (root / 'experiment.json').write_text(json.dumps({
        'question': 'Does appending four diagonal obstacle flags change Q-learning performance?',
        'changed_variable': 'observation width only: 8 versus 12',
        'controls': 'Same board, rewards, collision rules, learning parameters, episode budgets and paired seeds.',
        'evaluation': 'Frozen policies, epsilon 0. Same environment/action seeds for every policy.',
        'limitations': 'Three training seeds, one evaluation seed stream. Different actions consume environment '
                       'randomness differently, so corresponding episode IDs are not identical scenarios. '
                       'No hyperparameter tuning; this measures the existing training schedule.',
        'resumed_from': str(args.resume_from.resolve()) if previous else None,
        'training': 'Training budget is total games. Resumed runs retain saved epsilon and both RNG states; '
                    'their training CSV covers only the additional games, with a local rolling average.',
        'recording': f"Every {250 if previous else 50}th episode, full summary CSVs, checkpoints every 1000 games plus initial/final.",
        'runs': commands, 'results': summaries,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
