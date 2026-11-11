"""Rich CLI for Local-First RAG with Measured Quality."""
import sys
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import settings
from src.corpus_loader import CorpusLoader
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearchEngine
from src.hybrid_search import HybridSearchEngine
from src.evaluation import RAGEvaluator
from src.rag_engine import RAGEngine

console = Console()

def get_pipeline():
    """Initializes and returns the shared retrieval and RAG components."""
    loader = CorpusLoader(settings.corpus_dir)
    chunks = loader.load_corpus()
    vstore = VectorStore()
    if vstore.count() != len(chunks):
        console.print(f"[bold yellow]Indexing {len(chunks)} chunks in SQLite database...[/bold yellow]")
        vstore.index_chunks(chunks)
    kengine = KeywordSearchEngine(chunks)
    hybrid = HybridSearchEngine(vstore, kengine)
    rag = RAGEngine(hybrid)
    return chunks, vstore, kengine, hybrid, rag

def cmd_eval(args):
    console.print(Panel.fit("[bold cyan]🚀 Running Local-First RAG Quality Evaluation Benchmark...[/bold cyan]\n"
                            f"[dim]Measuring Recall@k, MRR, MAP, and NDCG across 36 test queries on Apple Silicon / Local CPU[/dim]"))
    evaluator = RAGEvaluator()
    report = evaluator.run_evaluation(alpha=args.alpha)

    # 1. Primary Metrics Table
    t1 = Table(title="📊 Primary Retrieval Benchmark: Recall@k & Ranking", header_style="bold magenta")
    t1.add_column("Search Mode", style="cyan", no_wrap=True)
    t1.add_column("Recall@1", justify="right")
    t1.add_column("Recall@3", justify="right")
    t1.add_column("Recall@5", justify="right")
    t1.add_column("Recall@10", justify="right")
    t1.add_column("MRR", justify="right", style="green")
    t1.add_column("MAP", justify="right")
    t1.add_column("NDCG@5", justify="right", style="yellow")
    t1.add_column("P50 Latency (ms)", justify="right")

    for mode_key, m in report.modes.items():
        t1.add_row(
            m.display_name,
            f"{m.recall_at_k.get(1, 0.0) * 100:.1f}%",
            f"{m.recall_at_k.get(3, 0.0) * 100:.1f}%",
            f"{m.recall_at_k.get(5, 0.0) * 100:.1f}%",
            f"{m.recall_at_k.get(10, 0.0) * 100:.1f}%",
            f"{m.mrr:.3f}",
            f"{m.map_score:.3f}",
            f"{m.ndcg_at_k.get(5, 0.0):.3f}",
            f"{m.latency_ms_p50:.2f} ms"
        )
    console.print(t1)

    # 2. Archetype Breakdown Table
    t2 = Table(title="🔬 Recall@5 Breakdown by Query Archetype", header_style="bold blue")
    t2.add_column("Search Mode", style="cyan")
    t2.add_column("Keyword Exact (12 queries)", justify="right")
    t2.add_column("Semantic Concept (10 queries)", justify="right")
    t2.add_column("Hybrid Balanced (14 queries)", justify="right")

    for mode_key, m in report.modes.items():
        kw = f"{m.category_recall_at_5.get('keyword_exact', 0.0) * 100:.1f}%"
        sem = f"{m.category_recall_at_5.get('semantic_concept', 0.0) * 100:.1f}%"
        hyb = f"{m.category_recall_at_5.get('hybrid_balanced', 0.0) * 100:.1f}%"
        t2.add_row(m.display_name, kw, sem, hyb)
    console.print(t2)

    # Save report
    out_path = Path(args.export) if args.export else settings.report_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    console.print(f"[bold green]✓ Benchmark report successfully exported to:[/bold green] [underline]{out_path}[/underline]")

def cmd_search(args):
    _, _, _, hybrid, _ = get_pipeline()
    console.print(f"[bold green]Searching query:[/bold green] '{args.query}' [dim](Mode: {args.mode}, Top-K: {args.top_k})[/dim]")
    results = hybrid.search(args.query, top_k=args.top_k, mode=args.mode, alpha=args.alpha)
    
    table = Table(title=f"Search Results ({args.mode})", header_style="bold cyan")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Chunk ID", style="magenta")
    table.add_column("Document / Section", style="white")
    
    for r in results:
        dense_info = f"D#{r.dense_rank}" if hasattr(r, "dense_rank") else ""
        bm25_info = f"B#{r.bm25_rank}" if hasattr(r, "bm25_rank") else ""
        rank_signals = f" ({dense_info} {bm25_info})" if (dense_info or bm25_info) else ""
        table.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            r.chunk_id,
            f"{r.doc_title} -> {r.section_title}{rank_signals}"
        )
    console.print(table)

def cmd_ask(args):
    _, _, _, _, rag = get_pipeline()
    console.print(f"[bold green]Query:[/bold green] {args.query} [dim](Mode: {args.mode})[/dim]\n")
    response = rag.ask(args.query, mode=args.mode, top_k=args.top_k, alpha=args.alpha)
    
    console.print(Panel(Markdown(response.answer), title="🤖 Grounded Local RAG Response", border_style="cyan"))
    
    t = Table(title=f"Sources Cited ({len(response.sources)} retrieved)", header_style="bold yellow")
    t.add_column("Rank", justify="center")
    t.add_column("Chunk ID", style="magenta")
    t.add_column("Document & Section", style="white")
    t.add_column("Score", justify="right")
    
    for s in response.sources:
        t.add_row(str(s.rank), s.chunk_id, f"{s.doc_title} -> {s.section_title}", f"{s.score:.4f}")
    console.print(t)
    console.print(f"[dim]Retrieval: {response.retrieval_latency_ms} ms | Synthesis: {response.synthesis_latency_ms} ms | Total: {response.total_latency_ms} ms[/dim]")

def cmd_ingest(args):
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else settings.corpus_dir
    loader = CorpusLoader(corpus_dir)
    chunks = loader.load_corpus()
    vstore = VectorStore()
    vstore.index_chunks(chunks, force_reindex=args.force)
    console.print(f"[bold green]✓ Ingested {len(chunks)} chunks into SQLite database at {vstore.db_path}[/bold green]")

def cmd_serve(args):
    import uvicorn
    console.print(f"[bold cyan]Starting Local-First RAG Dashboard on http://{args.host}:{args.port}[/bold cyan]")
    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=False)

def main():
    parser = argparse.ArgumentParser(description="Local-First RAG with Measured Quality")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run Recall@k quality benchmark across search modes")
    p_eval.add_argument("--alpha", type=float, default=settings.default_alpha, help="Convex alpha for linear hybrid fusion")
    p_eval.add_argument("--export", type=str, default=None, help="Output path for JSON report")

    # search
    p_search = subparsers.add_parser("search", help="Search the corpus")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("--mode", type=str, default="hybrid_rrf", choices=["hybrid_rrf", "hybrid_linear", "dense", "bm25"])
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--alpha", type=float, default=settings.default_alpha)

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question and generate an answer with citations")
    p_ask.add_argument("query", type=str, help="User question")
    p_ask.add_argument("--mode", type=str, default="hybrid_rrf", choices=["hybrid_rrf", "hybrid_linear", "dense", "bm25"])
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--alpha", type=float, default=settings.default_alpha)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest and index documents")
    p_ingest.add_argument("--corpus-dir", type=str, default=None)
    p_ingest.add_argument("--force", action="store_true", help="Force reindexing")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch the Web UI Dashboard")
    p_serve.add_argument("--host", type=str, default=settings.host)
    p_serve.add_argument("--port", type=int, default=settings.port)

    args = parser.parse_args()
    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)

if __name__ == "__main__":
    main()
