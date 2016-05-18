'use strict';

const should = require('should');
const SparseMatrix = require('..');

describe('Sparse Matrix', function () {
    it('mmul', function () {
        var m1 = new SparseMatrix([[2,0,1],[0,0,3]]);
        var m2 = new SparseMatrix([[0,1],[2,0],[0,0]]);
        var m3 = m1.mmul(m2);

        should(m1.cardinality).equal(3);
        should(m2.cardinality).equal(2);
        should(m3.cardinality).equal(1);

        should(m3.get(0, 1)).equal(2);
    });

    it('kronecker', function () {
        const matrix1 = new SparseMatrix([[1, 2], [3, 4]]);
        const matrix2 = new SparseMatrix([[0, 5], [6, 7]]);
        const product = matrix1.kroneckerProduct(matrix2);
        should(product.to2DArray()).eql([
            [0, 5, 0, 10],
            [6, 7, 12, 14],
            [0, 15, 0, 20],
            [18, 21, 24, 28]
        ]);
    });

    it('isSymmetric', function () {
        should((new SparseMatrix(10, 10)).isSymmetric()).true();
        should((new SparseMatrix(15, 10)).isSymmetric()).false();
        
        let m = new SparseMatrix([[0, 1], [1, 0]]);
        should(m.isSymmetric()).true();
        
        m = new SparseMatrix([[0, 1], [0, 1]]);
        should(m.isSymmetric()).false();
    });
});
