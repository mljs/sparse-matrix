import nodeResolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';

export default {
    entry: 'src/SparseMatrix.js',
    dest: 'dist/SparseMatrix.js',
    format: 'cjs',
    plugins: [
        nodeResolve({
            jsnext: true,
            main: true
        }),
        commonjs()
    ]
}
