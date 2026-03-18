import { BasicEvaluator, IRunnerPlugin } from "@sourceacademy/conductor/runner";
import { PyodideAPI } from "pyodide";
import { loadPyodideGeneric } from "./loadPyodide";

/**
 * Returns the set of top-level module roots for all import statements in the chunk.
 *
 * Examples:
 *   import torch.nn, numpy      → {"torch", "numpy"}
 *   from torch import *         → {"torch"}
 *   from torch.path import join → {"torch"}
 */
export function getImportRoots(chunk: string): Set<string> {
  const roots = new Set<string>();
  for (const rawLine of chunk.split(/\r?\n/)) {
    const commentIndex = rawLine.indexOf("#");
    const cleanLine = commentIndex >= 0 ? rawLine.slice(0, commentIndex) : rawLine;

    const m = cleanLine.match(/^\s*(?:import\s+(.+)|from\s+([\w.]+)\s+import\s+.+)$/);
    if (!m) continue;

    if (m[1]) {
      // import X, Y.Z as w, ...
      for (const part of m[1].split(",")) {
        const token = part.trim().replace(/\s+as\s+\w+$/i, "");
        const root = token.split(".")[0].trim();
        if (root && !root.startsWith(".")) roots.add(root);
      }
    } else {
      // from X.Y import ...
      const root = m[2].split(".")[0];
      if (root && !root.startsWith(".")) roots.add(root);
    }
  }
  return roots;
}

/**
 * Rewrites import statements for `moduleName` to use the pyodide-injected global
 * `__sa_import_<moduleName>`. Non-matching lines are passed through unchanged.
 *
 * Cases handled:
 *   import <mod>                         → <mod> = __sa_import_<mod>
 *   import <mod> as t                    → t = __sa_import_<mod>
 *   import a, <mod>, b                   → import a, b\n<mod> = __sa_import_<mod>
 *   import <mod>.sub [as y]              → y = __sa_import_<mod>.sub  (or <mod> = __sa_import_<mod> if no alias)
 *   from <mod>[.sub] import a [as x], b → x = __sa_import_<mod>[.sub].a\nb = __sa_import_<mod>[.sub].b
 *   from <mod> import *                  → globals() loop over dir(__sa_import_<mod>)
 */
export function hijackImports(
  chunk: string,
  moduleName: string
): { code: string; found: boolean } {
  const injected = `__sa_import_${moduleName}`;
  const lines = chunk.split(/\r?\n/);
  const output: string[] = [];
  let found = false;

  for (const rawLine of lines) {
    const commentIndex = rawLine.indexOf("#");
    const cleanLine = commentIndex >= 0 ? rawLine.slice(0, commentIndex) : rawLine;

    // --- Cases 1 & 2: `import ...` ---
    const importM = cleanLine.match(/^\s*import\s+(.+)$/);
    if (importM) {
      const kept: string[] = [];
      const replacements: string[] = [];

      for (const part of importM[1].split(",")) {
        const token = part.trim();
        if (!token) continue;

        const asMatch = token.match(/^([\w.]+)\s+as\s+(\w+)$/i);
        const name = asMatch ? asMatch[1] : token;
        const alias = asMatch ? asMatch[2] : null;
        const root = name.split(".")[0];

        if (root === moduleName) {
          found = true;
          const subpath = name.split(".").slice(1).join(".");
          if (alias) {
            // import torch.nn as F  →  F = __sa_import_torch.nn
            // import torch as t     →  t = __sa_import_torch
            replacements.push(subpath ? `${alias} = ${injected}.${subpath}` : `${alias} = ${injected}`);
          } else {
            // import torch.nn  →  torch = __sa_import_torch
            // import torch     →  torch = __sa_import_torch
            replacements.push(`${moduleName} = ${injected}`);
          }
        } else {
          kept.push(alias ? `${name} as ${alias}` : name);
        }
      }

      const resultLines: string[] = [];
      if (kept.length > 0) resultLines.push(`import ${kept.join(", ")}`);
      resultLines.push(...replacements);
      output.push(resultLines.join("\n"));
      continue;
    }

    // --- Cases 3 & 4: `from ... import ...` ---
    const fromM = cleanLine.match(/^\s*from\s+([\w.]+)\s+import\s+(.+)$/);
    if (fromM) {
      const fromModule = fromM[1];
      const fromRoot = fromModule.split(".")[0];

      if (fromRoot === moduleName) {
        found = true;
        const subpath = fromModule.split(".").slice(1).join(".");
        const base = subpath ? `${injected}.${subpath}` : injected;
        const importList = fromM[2].trim();

        if (importList === "*") {
          // from torch import *
          output.push(
            `for __sa_k in dir(${base}):\n    if not __sa_k.startswith('_'):\n        globals()[__sa_k] = getattr(${base}, __sa_k)`
          );
        } else {
          // from torch.nn import Linear, Conv2d as C
          const assignments: string[] = [];
          for (const part of importList.split(",")) {
            const token = part.trim();
            if (!token) continue;
            const asMatch = token.match(/^(\w+)\s+as\s+(\w+)$/i);
            if (asMatch) {
              assignments.push(`${asMatch[2]} = ${base}.${asMatch[1]}`);
            } else {
              assignments.push(`${token} = ${base}.${token}`);
            }
          }
          output.push(assignments.join("\n"));
        }
        continue;
      }
    }

    output.push(rawLine);
  }

  return { code: output.join("\n"), found };
}

export default class PythonEvaluator extends BasicEvaluator {
  pyodide: Promise<PyodideAPI>;
  private torchLoaded: boolean = false;

  constructor(conductor: IRunnerPlugin) {
    super(conductor);
    this.pyodide = loadPyodideGeneric().then(async pyodide => {
      await pyodide.loadPackage("micropip");
      await pyodide.setStdout({
        batched: (output: string) => {
          this.conductor.sendOutput(output);
        }
      });
      return pyodide;
    });
  }

  async evaluateChunk(chunk: string): Promise<void> {
    const pyodide = await this.pyodide;

    const roots = getImportRoots(chunk);
    const { code, found: needsTorch } = hijackImports(chunk, "torch");

    if (needsTorch && !this.torchLoaded) {
      await this.loadTorch(pyodide);
      pyodide.globals.set("__sa_import_torch", pyodide.globals.get("torch"));
      this.torchLoaded = true;
    }

    roots.delete("torch");
    if (roots.size > 0) {
      const modulesArray = Array.from(roots);
      const installerCode = `\nimport importlib, micropip\nmods = ${JSON.stringify(modulesArray)}\nmissing = []\nfor m in mods:\n    try:\n        importlib.import_module(m)\n    except Exception:\n        missing.append(m)\nif missing:\n    await micropip.install(missing)\n`;
      await pyodide.runPythonAsync(installerCode);
    }

    console.log("Executing final chunk in Pyodide:\n", code);
    const output = await pyodide.runPythonAsync(code);
    this.conductor.sendOutput(output);
  }

  private async loadTorch(pyodide: PyodideAPI) {
    console.log("Loading torch dependencies...");

    const urls = [
      "https://veehz.github.io/torch/build/browser/torch.browser.umd.js"
    ];

    try {
      if (typeof document !== 'undefined') {
        const loadScript = (url: string) => new Promise<void>((resolve, reject) => {
          const script = document.createElement("script");
          script.src = url;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
          document.head.appendChild(script);
        });
        for (const url of urls) {
          await loadScript(url);
        }
      } else if (typeof (globalThis as any).importScripts === 'function') {
        (globalThis as any).importScripts(...urls);
      } else {
        throw new Error("Neither 'document' nor 'importScripts' available to load torch dependencies.");
      }
    } catch (err) {
      console.error("Error loading torch scripts:", err);
      throw err;
    }

    const torchGlobal = (globalThis as any).torch;
    if (!torchGlobal) {
      throw new Error("torch global not found after loading scripts (check window.torch or globalThis.torch)");
    }

    pyodide.globals.set("js_torch", torchGlobal);

    try {
      const response = await fetch('https://veehz.github.io/torch/examples/pyodide/bridge.py');
      if (!response.ok) {
        throw new Error(`Failed to fetch bridge.py: ${response.status} ${response.statusText}`);
      }
      const bridgeCode = await response.text();
      await pyodide.runPythonAsync(bridgeCode);

      // Verify torch is defined
      const hasTorch = pyodide.runPython(" 'torch' in globals() ");
      if (!hasTorch) {
        console.warn("torch not found in globals after bridge.py");
      } else {
        console.log("Torch loaded successfully into Pyodide globals.");
      }
    } catch (err) {
      console.error("Error setting up torch bridge:", err);
      throw err;
    }
  }
}
