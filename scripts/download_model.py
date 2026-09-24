import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--allow-patterns", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError as e:
        print(f"ERROR: huggingface_hub not available: {e}")
        sys.exit(1)

    api = HfApi()
    info = api.model_info(args.repo, revision=args.revision)
    total = 0
    sizes = {}
    for s in info.siblings:
        sz = getattr(s, "size", None) or 0
        sizes[s.rfilename] = sz
        total += sz
    print(f"REPO {args.repo} rev={args.revision} total_bytes={total} ({total/1e9:.2f} GB)")
    for name, sz in sorted(sizes.items(), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {sz/1e6:.1f} MB")

    os.makedirs(args.local_dir, exist_ok=True)
    kwargs = dict(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.local_dir,
    )
    if args.allow_patterns:
        kwargs["allow_patterns"] = [p.strip() for p in args.allow_patterns.split(",")]

    path = snapshot_download(**kwargs)
    print(f"DONE path={path}")


if __name__ == "__main__":
    main()