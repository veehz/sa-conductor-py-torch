import { BasicEvaluator, IRunnerPlugin } from "@sourceacademy/conductor/runner";
import type { PyodideInterface } from "pyodide";
import { loadPyodideGeneric } from "./loadPyodide";
import { loadTorch } from "./loadTorch";
import { rewriteTorchImports, getNonTorchImportRoots } from "./importAnalyzer";

export default class PythonEvaluator extends BasicEvaluator {
  private pyodide: Promise<PyodideInterface>;
  private torchLoaded = false;

  constructor(conductor: IRunnerPlugin) {
    super(conductor);
    console.log("[PythonEvaluator] constructor called");
    this.pyodide = loadPyodideGeneric().then(async pyodide => {
      console.log("[PythonEvaluator] Pyodide loaded, loading micropip...");
      await pyodide.loadPackage("micropip");
      await pyodide.setStdout({
        batched: (output: string) => {
          this.conductor.sendOutput(output);
        },
      });
      console.log("[PythonEvaluator] Ready.");
      return pyodide;
    });
  }

  async evaluateChunk(chunk: string): Promise<void> {
    console.log("[PythonEvaluator] evaluateChunk called");
    const pyodide = await this.pyodide;

    // --- Use Python's ast module (via Pyodide) to detect and rewrite torch imports ---
    const { code, hasTorch } = await rewriteTorchImports(pyodide, chunk);
    console.log("[PythonEvaluator] needsTorch:", hasTorch, "torchLoaded:", this.torchLoaded);

    if (hasTorch && !this.torchLoaded) {
      console.log("[PythonEvaluator] loading torch...");
      await loadTorch(pyodide);
      pyodide.globals.set("__sa_import_torch", pyodide.globals.get("torch"));
      this.torchLoaded = true;
      console.log("[PythonEvaluator] torch loaded and __sa_import_torch set");
    }

    // --- Install any other imported modules via micropip ---
    const otherRoots = await getNonTorchImportRoots(pyodide, chunk);
    if (otherRoots.size > 0) {
      const modulesArray = Array.from(otherRoots);
      console.log("[PythonEvaluator] installing via micropip:", modulesArray);
      const installerCode = `
import importlib, micropip
mods = ${JSON.stringify(modulesArray)}
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    await micropip.install(missing)
`;
      await pyodide.runPythonAsync(installerCode);
    }

    // --- Execute the (possibly rewritten) code ---
    console.log("[PythonEvaluator] executing code in Pyodide");
    const output = await pyodide.runPythonAsync(code);
    console.log("[PythonEvaluator] execution complete, output:", output);
    this.conductor.sendOutput(output);
  }
}
