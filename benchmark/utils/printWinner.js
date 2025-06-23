export function printWinner(results) {
  // results: Array of { label: string, avg: number }
  if (!Array.isArray(results) || results.length < 2) {
    console.log('Need at least two results to compare.');
    return;
  }

  // Sort by avg ascending (fastest first)
  const sorted = [...results].sort((a, b) => a.avg - b.avg);
  const winner = sorted[0];

  console.log(
    `  -> ${winner.label} was the fastest (${winner.avg.toFixed(2)} ms avg)\n`,
  );

  for (let i = 1; i < sorted.length; i++) {
    const slower = sorted[i];
    const speedup = (slower.avg / winner.avg).toFixed(2);
    const percent = (((slower.avg - winner.avg) / slower.avg) * 100).toFixed(1);
    console.log(
      `     ${winner.label} was ${speedup}x faster (${percent}% faster) than ${
        slower.label
      } (${slower.avg.toFixed(2)} ms avg)`,
    );
  }
  console.log();
}
