#!/usr/bin/env python3
"""Read-only, sampled MS MARCO preprocessing audit.

Requires: ftfy, PostgreSQL running in the project's Docker container, psql in container.
Does not modify the database or indexes. Uses a single sampled COPY query.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import csv

from ftfy import fix_text, __version__ as ftfy_version


def ascii_tokens(text):
    """ASCII-alnum tokenizer matching the reviewed C++ SearchEngine::tokenize rules."""
    tokens = []
    buf = []
    for char in text:
        o = ord(char)
        if 65 <= o <= 90:
            buf.append(chr(o + 32))
        elif 97 <= o <= 122 or 48 <= o <= 57:
            buf.append(char)
        elif buf:
            tokens.append(''.join(buf))
            buf.clear()
    if buf:
        tokens.append(''.join(buf))
    return tokens


CONFIGS = {
    'encoding_focused': dict(unescape_html=False, fix_latin_ligatures=False, uncurl_quotes=False),
    'encoding_ligatures': dict(unescape_html=False, uncurl_quotes=False),
    'default': {},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-percent', type=float, default=1.2, help='SYSTEM sample percent (default 1.2, ~100k rows)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=120000, help='maximum sampled rows')
    p.add_argument('--container', default='hybrid-retrieval-db')
    p.add_argument('--db-user', default='retrieval')
    p.add_argument('--db-name', default='retrieval_db')
    p.add_argument('--output-dir', default='reports/data_quality')
    p.add_argument('--examples-per-group', type=int, default=8)
    args = p.parse_args()
    if not (0 < args.sample_percent <= 100):
        p.error('--sample-percent must be in (0, 100]')
    if args.limit <= 0 or args.examples_per_group < 0:
        p.error('--limit must be positive; --examples-per-group must be nonnegative')

    # Use PostgreSQL COPY for robust, tab-delimited transfer of base64 UTF-8 data.
    # A READ ONLY transaction ensures the query cannot write to database tables.
    sql = f"""BEGIN READ ONLY;
SET LOCAL statement_timeout = '20min';
COPY (
  SELECT
    document_id,
    translate(
      encode(convert_to(content, 'UTF8'), 'base64'),
      chr(10) || chr(13),
      ''
    )
  FROM msmarco_full_documents
  TABLESAMPLE SYSTEM ({args.sample_percent}) REPEATABLE ({args.seed})
  LIMIT {args.limit}
) TO STDOUT WITH (FORMAT text, DELIMITER E'\\t');
COMMIT;
"""
    command = ['docker', 'exec', '-i', args.container, 'psql', '-X', '-q', '-v', 'ON_ERROR_STOP=1',
               '-U', args.db_user, '-d', args.db_name]
    print('Starting read-only sample audit:', ' '.join(command), flush=True)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding='utf-8')
    proc.stdin.write(sql)
    proc.stdin.close()

    stats = {name: Counter() for name in CONFIGS}
    examples = {name: [] for name in CONFIGS}
    total = 0
    parse_errors = 0
    t0 = time.monotonic()
    for line in proc.stdout:
        line = line.rstrip('\r\n')
        if not line:
            continue
        try:
            docid, b64 = line.split('\t', 1)
            original = base64.b64decode(b64, validate=True).decode('utf-8')
            int(docid)
        except (ValueError, UnicodeError) as exc:
            parse_errors += 1
            if parse_errors <= 3:
                print('Malformed row:', str(exc), file=sys.stderr)
            continue
        total += 1
        before = ascii_tokens(original)
        for name, kwargs in CONFIGS.items():
            repaired = fix_text(original, **kwargs)
            st = stats[name]
            if repaired != original:
                st['changed_documents'] += 1
            after = ascii_tokens(repaired)
            if before != after:
                st['token_changed_documents'] += 1
                st['token_count_delta_sum'] += len(after) - len(before)
                if len(examples[name]) < args.examples_per_group:
                    examples[name].append({
                        'document_id': int(docid),
                        'original_excerpt': original[:400],
                        'repaired_excerpt': repaired[:400],
                        'tokens_before_excerpt': before[:60],
                        'tokens_after_excerpt': after[:60],
                    })
            if fix_text(repaired, **kwargs) != repaired:
                st['non_idempotent_documents'] += 1
            if abs(len(repaired) - len(original)) > 20:
                st['large_length_change_documents'] += 1
        if total % 20000 == 0:
            print(f'Processed {total:,} rows...', flush=True)
    stderr = proc.stderr.read()
    returncode = proc.wait()
    if returncode != 0:
        sys.exit(f'ERROR: psql exited with status {returncode}:\n{stderr}')
    if parse_errors:
        sys.exit(f'ERROR: {parse_errors} rows could not be parsed; results not written')
    if not total:
        sys.exit('ERROR: no sampled documents; increase --sample-percent')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, st in stats.items():
        results.append({
            'group': name,
            'sampled_documents': total,
            'changed_documents': st['changed_documents'],
            'changed_pct': round(100 * st['changed_documents'] / total, 4),
            'token_changed_documents': st['token_changed_documents'],
            'token_changed_pct': round(100 * st['token_changed_documents'] / total, 4),
            'token_count_delta_sum': st['token_count_delta_sum'],
            'non_idempotent_documents': st['non_idempotent_documents'],
            'large_length_change_documents': st['large_length_change_documents'],
        })
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'sample_method': 'PostgreSQL TABLESAMPLE SYSTEM; page-level sample, not uniform row sample',
        'sample_percent': args.sample_percent,
        'seed': args.seed,
        'limit': args.limit,
        'sampled_documents': total,
        'ftfy_version': ftfy_version,
        'tokenizer': 'Python ASCII-alnum tokenizer; source-level equivalence checked against C++ SearchEngine::tokenize; runtime parity not yet tested',
        'configs': CONFIGS,
        'results': results,
        'examples': examples,
        'runtime_seconds': round(time.monotonic() - t0, 2),
        'limitations': [
            'No database writes; no index rebuilds.',
            'ftfy configuration groups are nested, not independent factorial interventions.',
            'Token changes do not imply retrieval metric changes.',
            'SYSTEM sampling is by pages and may not represent all corpus subpopulations.',
            'Only a C++ tokenizer parity test can confirm equivalence to production BM25.',
        ],
    }
    with (out / 'preprocessing_audit.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (out / 'preprocessing_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (out / 'preprocessing_examples.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'document_id', 'original_excerpt', 'repaired_excerpt', 'tokens_before_excerpt', 'tokens_after_excerpt'])
        for name, items in examples.items():
            for e in items:
                writer.writerow([name, e['document_id'], e['original_excerpt'], e['repaired_excerpt'],
                                 json.dumps(e['tokens_before_excerpt'], ensure_ascii=False),
                                 json.dumps(e['tokens_after_excerpt'], ensure_ascii=False)])
    print('\n===== Controlled Preprocessing Audit =====')
    for r in results:
        print(f"{r['group']:20s} changed={r['changed_documents']:7,d} ({r['changed_pct']:7.4f}%) "
              f"token_changed={r['token_changed_documents']:6,d} ({r['token_changed_pct']:7.4f}%) "
              f"non_idempotent={r['non_idempotent_documents']}")
    print(f'Sampled documents: {total:,}; runtime: {report["runtime_seconds"]} s')
    print('Saved:', out / 'preprocessing_audit.json')
    print('Saved:', out / 'preprocessing_summary.csv')
    print('Saved:', out / 'preprocessing_examples.csv')


if __name__ == '__main__':
    main()
