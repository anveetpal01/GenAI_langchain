import { render } from "preact";
import "../shared/theme.css";
import "./sidepanel.css";

if (import.meta.env.DEV) await import("../dev/chrome-shim");

const { App } = await import("./App");
render(<App />, document.getElementById("app")!);
