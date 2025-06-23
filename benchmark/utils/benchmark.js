export function benchmark(fn, label, iterations = 5, logIt = true) {
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    fn();
    const t1 = performance.now();
    times.push(t1 - t0);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  if (logIt) {
    console.log(
      `${label}: avg ${avg.toFixed(2)} ms over ${iterations} runs \n`,
    );
  }
  return avg;
}
