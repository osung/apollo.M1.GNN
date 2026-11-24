"""Step 2 of the embedding + LLM matching baseline: LLM fit scoring.

Runs inside the vLLM docker image (see scripts/run_llm.sh) — it only needs
vllm/transformers plus the dependency-free src.baselines.llm_prompt module.

    scripts/run_llm.sh python3 scripts/llm_match_score.py --direction p2c
    scripts/run_llm.sh python3 scripts/llm_match_score.py --direction p2c \
        --shard-id 0 --num-shards 2          # one shard per machine

Results are appended to scores_<direction>_shard<i>of<n>.jsonl after every
chunk; re-running the same command resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.baselines.llm_prompt import MISSING, build_score_messages, parse_scores, strip_think


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


class VllmEngine:
    def __init__(self, model: str, gpu_util: float, max_len: int, max_seqs: int, max_tokens: int):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        log(f"loading {model} (gpu_util={gpu_util}, max_len={max_len}, max_seqs={max_seqs})")
        t0 = time.time()
        self.llm = LLM(model=model, trust_remote_code=True, dtype="auto",
                       gpu_memory_utilization=gpu_util, max_model_len=max_len,
                       max_num_seqs=max_seqs, enforce_eager=False)
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                                     stop=["<|im_end|>"])
        self.max_len, self.max_tokens = max_len, max_tokens
        log(f"model loaded in {time.time() - t0:.0f}s")

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # template without a thinking switch
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages_list: list[list[dict[str, str]]]) -> tuple[list[str], int, int]:
        prompts = [self._prompt(m) for m in messages_list]
        lens = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        keep = [i for i, n in enumerate(lens) if n + self.max_tokens <= self.max_len - 64]
        if len(keep) < len(prompts):
            log(f"  ! {len(prompts) - len(keep)} prompt(s) exceed the context window; unscored")
        outs = self.llm.generate([prompts[i] for i in keep], self.params, use_tqdm=False)
        texts = [""] * len(prompts)
        n_out = 0
        for i, o in zip(keep, outs, strict=True):
            texts[i] = strip_think(o.outputs[0].text or "")
            n_out += len(o.outputs[0].token_ids)
        return texts, sum(lens), n_out


class MockEngine:
    """Deterministic pseudo-scores; exercises the pipeline without a GPU or model."""

    def generate(self, messages_list: list[list[dict[str, str]]]) -> tuple[list[str], int, int]:
        texts = []
        for msgs in messages_list:
            user = msgs[-1]["content"]
            block = user.split(" 목록]\n", 1)[1].split("\n\n", 1)[0]
            results = []
            for line in block.split("\n"):
                head, _, body = line.partition(". ")
                if head.isdigit():
                    h = int(hashlib.md5(body.encode()).hexdigest(), 16) % 101
                    results.append({"id": int(head), "score": h, "reason": "mock"})
            texts.append(json.dumps({"results": results}, ensure_ascii=False))
        return texts, 0, 0


def _messages(job: dict[str, Any], positions: list[int]) -> list[dict[str, str]]:
    cands = [(k + 1, job["cand_texts"][pos]) for k, pos in enumerate(positions)]
    return build_score_messages(job["src_text"], job["src_label"], cands, job["cand_label"],
                                tech_side=job["tech_side"])


def score_chunk(
    engine: Any, jobs: list[dict[str, Any]], retry_batch: int
) -> tuple[list[dict], int, int]:
    """Score a chunk of jobs; candidates missing from a reply are retried once in smaller groups."""
    scores: list[dict[str, list]] = [{} for _ in jobs]
    work = [(j, list(range(len(job["cands"])))) for j, job in enumerate(jobs)]
    n_in = n_out = 0
    for rnd in range(2):
        if not work:
            break
        texts, ti, to = engine.generate([_messages(jobs[j], pos) for j, pos in work])
        n_in, n_out = n_in + ti, n_out + to
        retry = []
        for (j, positions), raw in zip(work, texts, strict=True):
            parsed = parse_scores(raw)
            miss = []
            for k, pos in enumerate(positions):
                if k + 1 in parsed:
                    scores[j][str(jobs[j]["cands"][pos])] = list(parsed[k + 1])
                else:
                    miss.append(pos)
            if rnd == 0:
                retry += [(j, miss[b : b + retry_batch]) for b in range(0, len(miss), retry_batch)]
        work = retry
    records = []
    for job, sc in zip(jobs, scores, strict=True):
        missing = [c for c in job["cands"] if str(c) not in sc]
        for c in missing:
            sc[str(c)] = list(MISSING)
        records.append({"job_id": job["job_id"], "direction": job["direction"],
                        "query": job["query"], "scores": sc, "missing": missing})
    return records, n_in, n_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/llm_match.yaml")
    parser.add_argument("--direction", default="p2c", choices=["p2c", "c2p"])
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=None,
                        help="score only jobs up to this candidate depth (job field 'depth')")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="score only the first N queries of this shard (speed test)")
    parser.add_argument("--model", default=None, help="override llm.model")
    parser.add_argument("--gpu-mem-util", type=float, default=None)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg["llm"]
    out_dir = Path(cfg["output_dir"])
    jobs_path = out_dir / f"jobs_{args.direction}.jsonl"
    out_path = out_dir / (
        f"scores_{args.direction}_shard{args.shard_id}of{args.num_shards}.jsonl"
    )

    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["job_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    jobs = []
    with open(jobs_path, encoding="utf-8") as f:
        for line in f:
            job = json.loads(line)
            if job["query_rank"] % args.num_shards != args.shard_id:
                continue
            if args.max_depth is not None and job.get("depth", 0) > args.max_depth:
                continue
            jobs.append(job)
    if args.limit_queries is not None:
        ranks = sorted({j["query_rank"] for j in jobs})[: args.limit_queries]
        jobs = [j for j in jobs if j["query_rank"] in set(ranks)]
    todo = [j for j in jobs if j["job_id"] not in done]
    log(f"{jobs_path.name}: shard {args.shard_id}/{args.num_shards} has {len(jobs):,} jobs, "
        f"{len(jobs) - len(todo):,} already done, {len(todo):,} to score -> {out_path}")
    if not todo:
        return

    if args.backend == "mock":
        engine: Any = MockEngine()
    else:
        cap = float(llm_cfg["gpu_mem_util_cap"])
        util = float(args.gpu_mem_util or llm_cfg["gpu_mem_util"])
        if util > cap:
            log(f"gpu_mem_util {util} clamped to policy cap {cap}")
            util = cap
        engine = VllmEngine(args.model or llm_cfg["model"], util, int(llm_cfg["max_model_len"]),
                            int(llm_cfg["max_num_seqs"]), int(llm_cfg["max_tokens"]))

    chunk = int(llm_cfg["chunk"])
    t0 = time.time()
    n_done = n_missing = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), chunk):
            part = todo[s : s + chunk]
            t1 = time.time()
            records, n_in, n_out = score_chunk(engine, part, int(llm_cfg["retry_batch"]))
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_missing += len(rec["missing"])
            f.flush()
            n_done += len(part)
            dt, el = time.time() - t1, time.time() - t0
            eta = el / n_done * (len(todo) - n_done)
            log(f"  {n_done:,}/{len(todo):,} jobs  {dt / len(part):.2f}s/job  "
                f"in {n_in // max(len(part), 1)}tok/job out {n_out / max(dt, 1e-6):.0f}tok/s  "
                f"missing {n_missing}  ETA {eta / 3600:.1f}h")
    log(f"done: {n_done:,} jobs in {(time.time() - t0) / 3600:.2f}h, "
        f"{n_missing} candidates without an LLM answer (scored 0)")


if __name__ == "__main__":
    main()
