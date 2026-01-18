import { BasicEvaluator, IRunnerPlugin } from "@sourceacademy/conductor";
import { PyodideAPI } from "pyodide";
import { loadPyodideGeneric } from "./loadPyodide";

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
      
      // import packages via micropip installation
      const importedPackageRoots = new Set<string>();
      const lines = chunk.split(/\r?\n/);
      const filteredLines: string[] = [];
      let torchImported = false;

      for (let rawLine of lines) {
        const line = rawLine.trim();
        if (line === "import torch") {
          torchImported = true;
          continue;
        }

        const commentIndex = rawLine.indexOf("#");
        const cleanLine = commentIndex >= 0 ? rawLine.slice(0, commentIndex) : rawLine;

        const importLike = cleanLine.match(/^\s*(?:import\s+(.+)|from\s+([\w\.]+)\s+import\s+(.+))$/);
        if (importLike) {
          if (importLike[1]) {
            const modulesPart = importLike[1];
            for (const part of modulesPart.split(",")) {
              const token = part.trim();
              if (!token) continue;
              const noAlias = token.replace(/\s+as\s+\w+$/i, "");
              const root = noAlias.split(".")[0].trim();
              if (root && !root.startsWith(".")) {
                importedPackageRoots.add(root);
              }
            }
          } else {
            const pkg = importLike[2].trim();
            if (!pkg.startsWith(".")) {
              const root = pkg.split(".")[0];
              if (root) importedPackageRoots.add(root);
            }
          }
        }
        filteredLines.push(rawLine);
      }

      if (torchImported && !this.torchLoaded) {
        await this.loadTorch(pyodide);
        this.torchLoaded = true;
      }

      if (importedPackageRoots.size > 0) {
        const modulesArray = Array.from(importedPackageRoots);
        const installerCode = `\nimport importlib, micropip\nmods = ${JSON.stringify(modulesArray)}\nmissing = []\nfor m in mods:\n    try:\n        importlib.import_module(m)\n    except Exception:\n        missing.append(m)\nif missing:\n    await micropip.install(missing)\n`;
        await pyodide.runPythonAsync(installerCode);
      }

      const finalChunk = filteredLines.join("\n");
      console.log("Executing final chunk in Pyodide:\n", finalChunk);
      const output = await pyodide.runPythonAsync(finalChunk);
      this.conductor.sendOutput(output);
    }

    private async loadTorch(pyodide: PyodideAPI) {
      console.log("Loading torch dependencies...");

      const urls = [
        "https://veehz.github.io/torch/vendor/gpu-browser.min.js",
        "https://veehz.github.io/torch/build/browser/torch.browser.umd.js"
      ];

      try {
        if (typeof document !== 'undefined') {
          const loadScript = (url: string) => new Promise<void>((resolve, reject) => {
            const script = document.createElement("script");
            script.src = url;
            script.onload = () => resolve();
            script.onerror = (e) => reject(new Error(`Failed to load script: ${url}`));
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

      // Setup torch_utils
      const torchGlobal = (globalThis as any).torch;
      if (!torchGlobal) {
        throw new Error("torch global not found after loading scripts (check window.torch or globalThis.torch)");
      }
      
      const { Tensor } = torchGlobal;
      const torch_utils = {
          get_data_from_tensor: function (tensor: any) {
              return tensor.data;
          },
          create_tensor_from_python_data: function (data: any, requires_grad: boolean) {
              const jsData = typeof data.toJs === "function" ? data.toJs() : data;
              return new Tensor(jsData, { requires_grad });
          },
          to_js_list: function (data: any) {
              return data.toJs();
          },
      };

      pyodide.globals.set("js_torch", torchGlobal);
      pyodide.globals.set("torch_utils", torch_utils);

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
