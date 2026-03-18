import nodeResolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import typescript from '@rollup/plugin-typescript';
import polyfillNode from 'rollup-plugin-polyfill-node';
import copy from 'rollup-plugin-copy';

export default [{
    input: 'src/index.ts',
    output: {
      file: 'dist/index.js',
      format: 'iife',
      inlineDynamicImports: true,
    },
    plugins: [typescript(), nodeResolve(), commonjs(), polyfillNode(), copy({ targets: [{ src: 'src/directory.json', dest: 'dist' }] })]
  }, {
    input: 'src/PythonEvaluator.ts',
    output: {
      file: 'dist/pyscript-evaluator.cjs',
      format: 'cjs',
      inlineDynamicImports: true,
    },
    plugins: [typescript(), nodeResolve({browser:true}), commonjs(), polyfillNode()]
  }];