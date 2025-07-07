import { defineConfig, globalIgnores } from 'eslint/config';
import ts from 'eslint-config-cheminfo-typescript/base';

export default defineConfig(globalIgnores(['benchmark/old', 'lib']), ts, {
  files: ['benchmark/**'],
  rules: {
    camelcase: 'off',
    'func-names': 'off',
    'no-console': 'off',
    'no-unused-vars': 'off',
  },
});
