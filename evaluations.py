import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

EVAL_CSV_PATH = os.path.join(os.path.dirname(__file__), 'evaluation_set', 'evaluation_set.csv')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
EVAL_REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'evaluations')



@dataclass
class EvalRow:
    full_name: str
    phase_type: str  # 'ACTIVE_DATE' | 'END_OF_LIFE_DATE'
    target_date_raw: str  # 'YYYY-MM-DD' or 'NOT_FOUND'
    status: str


@dataclass
class OutputRow:
    component: str
    active_date: Optional[str]
    eos_date: Optional[str]


@dataclass
class Comparison:
    component: str
    phase_type: str
    target_date: Optional[date]  # None if NOT_FOUND
    predicted_date: Optional[date]
    predicted_other_date: Optional[date]
    exact_match: bool
    exact_nf_match: bool
    wrong_phase_match: bool
    abs_days_delta: Optional[int]


@dataclass
class Metrics:
    # Totals
    csv_rows: int
    evaluated_rows: int  # total components evaluated (matched)
    targets_null: int    # reference NULL target dates (NOT_FOUND)
    outputs_null: int    # predicted NULL in outputs
    targets_with_date: int
    outputs_with_date: int
    # Exact
    exact_matches: int
    exact_nf_matches: int
    exact_date_matches: int
    # Mismatches
    mismatches_total: int
    mismatched_dates: int
    mismatches_pred_null_target_date: int
    mismatches_pred_date_target_null: int
    # Delta buckets (non-zero only)
    delta_buckets: Dict[str, int]


# -----------------------------
# Utilities
# -----------------------------

def normalize_name(name: str) -> str:
    """Normalize component names for matching between CSV and outputs."""
    if name is None:
        return ''
    s = name.strip()
    s = re.sub(r'^\d+\s*', '', s)  # drop leading index numbers if present (e.g., "1     Foo")
    s = re.sub(r'\s+', ' ', s)  # collapse whitespace
    return s.lower()


def parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse YYYY-MM-DD or return None for missing/NOT_FOUND/invalid."""
    if not value:
        return None
    s = str(value).strip()
    if s.upper() == 'NOT_FOUND':
        return None
    try:
        return datetime.strptime(s[:10], '%Y-%m-%d').date()
    except Exception:
        return None


# -----------------------------
# Evaluator
# -----------------------------

class LifecycleEvaluator:
    def __init__(self, evaluation_csv_path: str = EVAL_CSV_PATH, outputs_dir: str = OUTPUTS_DIR) -> None:
        self.evaluation_csv_path = evaluation_csv_path
        self.outputs_dir = outputs_dir

    # -------- I/O --------

    def _load_outputs(self, output_filename: str) -> Dict[str, OutputRow]:
        path = output_filename
        if not os.path.isabs(path):
            path = os.path.join(self.outputs_dir, output_filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results: Dict[str, OutputRow] = {}
        for item in data.get('results', []):
            component_name = item.get('component') or ''
            key = normalize_name(component_name)
            results[key] = OutputRow(
                component=component_name,
                active_date=item.get('active_date'),
                eos_date=item.get('eos_date'),
            )
        return results

    def _load_eval_rows(self) -> List[EvalRow]:
        rows: List[EvalRow] = []
        with open(self.evaluation_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(EvalRow(
                    full_name=(r.get('full_name') or '').strip(),
                    phase_type=(r.get('type') or '').strip().upper(),
                    target_date_raw=(r.get('target_date') or '').strip(),
                    status=(r.get('status') or '').strip(),
                ))
        return rows

    # -------- core comparison logic --------

    def _compare_row(self, eval_row: EvalRow, out_row: Optional[OutputRow]) -> Optional[Comparison]:
        if out_row is None:
            return None

        expected_field = 'active_date' if eval_row.phase_type == 'ACTIVE_DATE' else 'eos_date'
        other_field = 'eos_date' if expected_field == 'active_date' else 'active_date'

        expected_pred_raw = getattr(out_row, expected_field)
        other_pred_raw = getattr(out_row, other_field)

        target_dt = parse_iso_date(eval_row.target_date_raw)
        expected_pred_dt = parse_iso_date(expected_pred_raw)
        other_pred_dt = parse_iso_date(other_pred_raw)

        exact_nf_match = eval_row.target_date_raw.strip().upper() == 'NOT_FOUND' and expected_pred_dt is None
        exact_date_match = (target_dt is not None and expected_pred_dt is not None and target_dt == expected_pred_dt)

        # Wrong-phase if the "other" field equals target while expected is missing or different
        wrong_phase_match = (
            target_dt is not None and
            other_pred_dt is not None and
            target_dt == other_pred_dt and
            (expected_pred_dt is None or expected_pred_dt != target_dt)
        )

        abs_days_delta: Optional[int] = None
        if target_dt is not None and expected_pred_dt is not None:
            abs_days_delta = abs((expected_pred_dt - target_dt).days)

        return Comparison(
            component=out_row.component,
            phase_type=eval_row.phase_type,
            target_date=target_dt,
            predicted_date=expected_pred_dt,
            predicted_other_date=other_pred_dt,
            exact_match=(exact_nf_match or exact_date_match),
            exact_nf_match=exact_nf_match,
            wrong_phase_match=wrong_phase_match,
            abs_days_delta=abs_days_delta,
        )

    # -------- aggregation & reporting --------

    def _bucketize_nonzero(self, values: List[int]) -> Dict[str, int]:
        buckets = {
            '<= 1 day': 0,
            '<= 3 days': 0,
            '<= 7 days': 0,
            '<= 30 days': 0,
            '<= 90 days': 0,
            '> 90 days': 0,
        }
        for v in values:
            if v <= 0:
                continue
            if v <= 1:
                buckets['<= 1 day'] += 1
            elif v <= 3:
                buckets['<= 3 days'] += 1
            elif v <= 7:
                buckets['<= 7 days'] += 1
            elif v <= 30:
                buckets['<= 30 days'] += 1
            elif v <= 90:
                buckets['<= 90 days'] += 1
            else:
                buckets['> 90 days'] += 1
        return buckets

    def _aggregate(self, comparisons: List[Comparison], total_rows: int) -> Metrics:
        evaluated_rows = len(comparisons)

        targets_null = sum(1 for c in comparisons if c.target_date is None)
        outputs_null = sum(1 for c in comparisons if c.predicted_date is None)
        targets_with_date = evaluated_rows - targets_null
        outputs_with_date = evaluated_rows - outputs_null

        exact_nf_matches = sum(1 for c in comparisons if c.target_date is None and c.predicted_date is None)
        exact_date_matches = sum(1 for c in comparisons if (c.target_date is not None and c.predicted_date is not None and c.target_date == c.predicted_date))
        exact_matches = exact_nf_matches + exact_date_matches

        mismatches_pred_null_target_date = sum(1 for c in comparisons if (c.target_date is not None and c.predicted_date is None))
        mismatches_pred_date_target_null = sum(1 for c in comparisons if (c.target_date is None and c.predicted_date is not None))
        mismatched_dates = sum(1 for c in comparisons if (c.target_date is not None and c.predicted_date is not None and c.target_date != c.predicted_date))
        mismatches_total = mismatches_pred_null_target_date + mismatches_pred_date_target_null + mismatched_dates

        deltas_nonzero = [c.abs_days_delta for c in comparisons if c.abs_days_delta is not None and c.abs_days_delta > 0]
        delta_buckets = self._bucketize_nonzero(deltas_nonzero)

        return Metrics(
            csv_rows=total_rows,
            evaluated_rows=evaluated_rows,
            targets_null=targets_null,
            outputs_null=outputs_null,
            targets_with_date=targets_with_date,
            outputs_with_date=outputs_with_date,
            exact_matches=exact_matches,
            exact_nf_matches=exact_nf_matches,
            exact_date_matches=exact_date_matches,
            mismatches_total=mismatches_total,
            mismatched_dates=mismatched_dates,
            mismatches_pred_null_target_date=mismatches_pred_null_target_date,
            mismatches_pred_date_target_null=mismatches_pred_date_target_null,
            delta_buckets=delta_buckets,
        )

    @staticmethod
    def _format_table(rows: List[Tuple[str, str]], headers: Tuple[str, str]) -> str:
        col1_width = max(len(headers[0]), *(len(r[0]) for r in rows)) if rows else len(headers[0])
        col2_width = max(len(headers[1]), *(len(r[1]) for r in rows)) if rows else len(headers[1])
        sep = '+-' + '-' * col1_width + '-+-' + '-' * col2_width + '-+'
        lines = [
            sep,
            '| ' + headers[0].ljust(col1_width) + ' | ' + headers[1].rjust(col2_width) + ' |',
            sep,
        ]
        for k, v in rows:
            lines.append('| ' + k.ljust(col1_width) + ' | ' + v.rjust(col2_width) + ' |')
        lines.append(sep)
        return '\n'.join(lines)

    @staticmethod
    def _pct(a: int, b: int) -> float:
        return (100.0 * a / b) if b else 0.0

    def _derive_batch_id(self, output_file_arg: str) -> str:
        fname = os.path.basename(output_file_arg)
        m = re.match(r'catalog_research_results_(\d{8})_(\d{6})\.json', fname)
        if m:
            return m.group(2)
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _build_report_text(self, output_file_arg: str, m: Metrics) -> str:
        lines: List[str] = []
        lines.append('\nEvaluation run')
        lines.append('==============')
        lines.append(f"Evaluation CSV: {self.evaluation_csv_path}")
        out_disp = output_file_arg if os.path.isabs(output_file_arg) else os.path.join(self.outputs_dir, output_file_arg)
        lines.append(f"Outputs JSON  : {out_disp}")
        lines.append(f"Rows in CSV: {m.csv_rows}")
        lines.append('')
        overview_rows = [
            ("Total components evaluated", f"{m.evaluated_rows}"),
            ("Total reference NULL target dates", f"{m.targets_null}"),
            ("Total outputs NULL dates", f"{m.outputs_null}"),
            ("Total targets with date", f"{m.targets_with_date}"),
            ("Total outputs with date", f"{m.outputs_with_date}"),
        ]
        lines.append(self._format_table(overview_rows, ("Metric", "Count")))
        lines.append('')
        phase_rows = [
            ("Exact matches (overall)", f"{m.exact_matches} ({self._pct(m.exact_matches, m.evaluated_rows):.1f}%)"),
            ("  • NOT_FOUND aligned", f"{m.exact_nf_matches} ({self._pct(m.exact_nf_matches, m.evaluated_rows):.1f}%)"),
            ("  • Exact date matches", f"{m.exact_date_matches} ({self._pct(m.exact_date_matches, m.evaluated_rows):.1f}%)"),
            ("Mismatches (overall)", f"{m.mismatches_total} ({self._pct(m.mismatches_total, m.evaluated_rows):.1f}%)"),
            ("  • Mismatched dates", f"{m.mismatched_dates}"),
            ("  • Mismatch: predicted NULL, target has date", f"{m.mismatches_pred_null_target_date}"),
            ("  • Mismatch: predicted date present, target NULL", f"{m.mismatches_pred_date_target_null}"),
        ]
        lines.append(self._format_table(phase_rows, ("Phase Metrics", "Value")))
        lines.append('')
        bucket_rows = [(k, str(v)) for k, v in m.delta_buckets.items()]
        lines.append(self._format_table(bucket_rows, ("Date Delta Buckets (non-exact)", "Count")))
        return '\n'.join(lines)

    def _save_report(self, report_text: str, output_file_arg: str) -> str:
        os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
        batch_id = self._derive_batch_id(output_file_arg)
        filename = f"evaluation_metrics_{batch_id}.txt"
        path = os.path.join(EVAL_REPORTS_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        return path

    # -------- main entry --------

    def run(self, output_file: str) -> None:
        outputs_map = self._load_outputs(output_file)
        eval_rows = self._load_eval_rows()

        comparisons: List[Comparison] = []
        for r in eval_rows:
            key = normalize_name(r.full_name)
            out_row = outputs_map.get(key)
            cmp_res = self._compare_row(r, out_row)
            if cmp_res is None:
                continue
            comparisons.append(cmp_res)

        metrics = self._aggregate(comparisons, total_rows=len(eval_rows))
        report_text = self._build_report_text(output_file, metrics)
        print(report_text)
        saved_path = self._save_report(report_text, output_file)
        print(f"\nSaved evaluation report to: {saved_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate lifecycle date extraction against evaluation_set.csv')
    parser.add_argument('output_file', help="Filename in 'outputs/' or absolute path to the JSON batch results")
    args = parser.parse_args()

    evaluator = LifecycleEvaluator()
    evaluator.run(args.output_file)


if __name__ == '__main__':
    main() 