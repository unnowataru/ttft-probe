#!/usr/bin/env python3
import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import http.client

@dataclass
class Result:
    idx: int
    http_status: int
    ttft_s: float | None
    total_s: float | None
    error: str

def build_payload(model: str, max_tokens: int, idx: int) -> bytes:
    prompt = (
        f"[TTFT-TEST idx={idx}]\n"
        "日本語で, ITインフラ運用の観点から箇条書きで説明してください.\n"
        "条件:\n"
        "1) 1行に1項目\n"
        "2) 150項目\n"
        "3) 1行は20文字以上\n"
        "余計な前置きは不要.\n"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    return json.dumps(payload).encode("utf-8")

def one_request(base_url: str, model: str, max_tokens: int, idx: int, start_evt: threading.Event, timeout_s: int) -> Result:
    u = urlparse(base_url)
    host = u.hostname
    port = u.port or (443 if u.scheme == "https" else 80)
    prefix = (u.path or "").rstrip("/")
    path = prefix + "/chat/completions"

    start_evt.wait()
    t0 = time.perf_counter()
    ttft = None

    try:
        conn = http.client.HTTPSConnection(host, port, timeout=timeout_s) if u.scheme == "https" else http.client.HTTPConnection(host, port, timeout=timeout_s)

        body = build_payload(model, max_tokens, idx)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        status = resp.status

        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()

            if data == b"[DONE]":
                total = time.perf_counter() - t0
                conn.close()
                return Result(idx=idx, http_status=status, ttft_s=ttft, total_s=total, error="")

            if ttft is None:
                ttft = time.perf_counter() - t0

        total = time.perf_counter() - t0
        conn.close()
        return Result(idx=idx, http_status=status, ttft_s=ttft, total_s=total, error="stream ended unexpectedly")

    except Exception as e:
        total = time.perf_counter() - t0
        return Result(idx=idx, http_status=0, ttft_s=ttft, total_s=total, error=str(e))

def run(base_url: str, model: str, n: int, max_tokens: int, out_prefix: str, out_dir: str, timeout_s: int):
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = Path(out_dir) / f"{out_prefix}_{ts}.csv"

    start_evt = threading.Event()
    results: list[Result] = []

    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(one_request, base_url, model, max_tokens, i + 1, start_evt, timeout_s) for i in range(n)]
        time.sleep(0.2)
        start_evt.set()
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r.idx)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "http_status", "ttft_s", "total_s", "error"])
        for r in results:
            w.writerow([r.idx, r.http_status, r.ttft_s, r.total_s, r.error])

    ok = [r for r in results if r.http_status == 200 and r.ttft_s is not None]
    if ok:
        ttfts = sorted(r.ttft_s for r in ok if r.ttft_s is not None)
        p50 = ttfts[len(ttfts) // 2]
        p90 = ttfts[int(len(ttfts) * 0.9) - 1]
        mx = max(ttfts)
        print(f"saved: {out_path}")
        print(f"n={n}, max_tokens={max_tokens}, ok={len(ok)}/{len(results)}")
        print(f"TTFT p50={p50:.3f}s p90={p90:.3f}s max={mx:.3f}s")
    else:
        print(f"saved: {out_path}")
        print("no successful TTFT samples")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)  # e.g. http://10.10.23.32:8000/v1
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)  # 推奨: 512
    ap.add_argument("--out-prefix", required=True)          # e.g. n32, n33
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    run(args.base_url, args.model, args.n, args.max_tokens, args.out_prefix, args.out_dir, args.timeout)

if __name__ == "__main__":
    main()
