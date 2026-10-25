import { render } from "preact";
import "../shared/theme.css";
import "./options.css";

if (import.meta.env.DEV) await import("../dev/chrome-shim");

const { Options } = await import("./Options");
render(<Options />, document.getElementById("app")!);
