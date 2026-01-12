import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  docsSidebar: [
    "introduction",
    "quickstart",
    {
      type: "category",
      label: "Tutorial",
      items: [
        "tutorial/setup",
        "tutorial/web-ui",
        "tutorial/structured-io",
        "tutorial/interactive-app",
        "tutorial/interactive-tool",
      ],
    },
    {
      type: "category",
      label: "Concepts",
      items: ["concepts/architecture", "concepts/instrumentation"],
    },
    {
      type: "link",
      label: "Examples",
      href: "https://github.com/yiouli/pixie-examples",
    },
    {
      type: "category",
      label: "API Reference",
      items: ["api/overview"],
    },
  ],
};

export default sidebars;
