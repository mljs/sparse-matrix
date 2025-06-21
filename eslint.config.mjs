import { defineConfig, globalIgnores } from 'eslint/config';
import ts from 'eslint-config-cheminfo-typescript/base';

export default defineConfig(globalIgnores(['benchmark/class/*', 'lib']), ts, {
  files: ['benchmark/**'],
  rules: {
    'no-console': 'off',
    'no-unused-vars': 'off',
  },
});
