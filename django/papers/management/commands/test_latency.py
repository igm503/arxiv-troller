import time
import statistics
from django.core.management.base import BaseCommand
from django.db import connection
from papers.models import Paper, Embedding
from papers.views import get_similar_embeddings, get_date_cutoff


class Command(BaseCommand):
    help = "Test latency of single paper search across different date ranges"

    def add_arguments(self, parser):
        parser.add_argument(
            "--paper-id",
            type=int,
            help="Specific paper ID to test with (default: random paper with embedding per trial)",
        )
        parser.add_argument(
            "--num-trials",
            type=int,
            default=5,
            help="Number of trials per date range (default: 5)",
        )
        parser.add_argument(
            "--num-results",
            type=int,
            default=40,
            help="Number of results to fetch (default: 40)",
        )

    def handle(self, *args, **options):
        # Set HNSW parameters
        with connection.cursor() as cursor:
            cursor.execute("SET hnsw.ef_search = 64")
            cursor.execute("SET hnsw.iterative_scan = 'relaxed_order'")
            cursor.execute("SET hnsw.max_scan_tuples = 4000")

        # Get test papers
        paper_id = options["paper_id"]
        if paper_id:
            test_paper = Paper.objects.get(id=paper_id)
            if not Embedding.objects.filter(paper=test_paper).exists():
                self.stdout.write(self.style.ERROR(f"Paper {paper_id} has no embedding"))
                return
            test_papers = [test_paper] * options["num_trials"]
            self.stdout.write(f"\nTesting with fixed paper: {test_paper.title[:60]}...")
            self.stdout.write(f"Paper ID: {test_paper.id}")
        else:
            # Get random papers with embeddings for each trial
            test_papers = list(
                Paper.objects.filter(embedding__isnull=False).order_by("?")[: options["num_trials"]]
            )
            if len(test_papers) < options["num_trials"]:
                self.stdout.write(
                    self.style.WARNING(
                        f"Only found {len(test_papers)} papers with embeddings, "
                        f"requested {options['num_trials']} trials"
                    )
                )
            if not test_papers:
                self.stdout.write(self.style.ERROR("No papers with embeddings found"))
                return
            self.stdout.write(f"\nTesting with {len(test_papers)} random papers (one per trial)")

        self.stdout.write(f"Trials per date range: {len(test_papers)}")
        self.stdout.write(f"Results per query: {options['num_results']}\n")

        # Date ranges to test (from 1 month onwards)
        date_ranges = ["1month", "3months", "6months", "1year", "2years", "all"]

        results = {}

        for date_filter in date_ranges:
            self.stdout.write(f"\nTesting date range: {date_filter}")
            self.stdout.write("-" * 80)

            # Build the valid papers query for this date range
            date_cutoff = get_date_cutoff(date_filter)

            # Get base query (we'll exclude current paper in each trial)
            base_paper_query = Paper.objects.all()
            if date_cutoff:
                base_paper_query = base_paper_query.filter(created__gte=date_cutoff)

            # Count eligible papers (approximate, will vary slightly per trial)
            eligible_count = base_paper_query.count()
            self.stdout.write(f"Eligible papers in range: {eligible_count:,}")

            if eligible_count == 0:
                self.stdout.write(self.style.WARNING("No papers in this range, skipping"))
                continue

            # Run trials
            trial_times = []

            for trial_idx, paper in enumerate(test_papers):
                # Exclude current paper from results
                paper_query = base_paper_query.exclude(id=paper.id)

                start = time.time()
                similar_papers = get_similar_embeddings(paper, paper_query, options["num_results"])
                elapsed = (time.time() - start) * 1000  # Convert to ms

                trial_times.append(elapsed)

                self.stdout.write(
                    f"  Trial {trial_idx + 1} (paper {paper.id}): {elapsed:.2f}ms "
                    f"({len(similar_papers)} results)"
                )

            # Calculate statistics
            results[date_filter] = {
                "eligible_papers": eligible_count,
                "times": trial_times,
                "mean": statistics.mean(trial_times),
                "median": statistics.median(trial_times),
                "min": min(trial_times),
                "max": max(trial_times),
                "stdev": statistics.stdev(trial_times) if len(trial_times) > 1 else 0,
            }

        # Print summary
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("SUMMARY")
        self.stdout.write("=" * 80)
        self.stdout.write(
            f"{'Date Range':<12} {'Papers':<12} {'Mean':<10} {'Median':<10} "
            f"{'Min':<10} {'Max':<10} {'StdDev':<10}"
        )
        self.stdout.write("-" * 80)

        for date_filter in date_ranges:
            if date_filter not in results:
                continue
            r = results[date_filter]
            self.stdout.write(
                f"{date_filter:<12} {r['eligible_papers']:<12,} "
                f"{r['mean']:<10.2f} {r['median']:<10.2f} "
                f"{r['min']:<10.2f} {r['max']:<10.2f} "
                f"{r['stdev']:<10.2f}"
            )
