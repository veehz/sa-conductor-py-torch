import { initialise } from "@sourceacademy/conductor/runner";
import PythonEvaluator from "./PythonEvaluator";

const {runnerPlugin, conduit} = initialise(PythonEvaluator);