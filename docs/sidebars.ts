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
      label: "Guides",
      items: [
        "guides/register-your-application",
        "guides/run-local-server-with-options",
        "guides/app-names-and-descriptions",
        "guides/use-structured-io",
        "guides/make-your-application-interactive",
        "guides/interactivity-inside-tools",
        "guides/enable-tracing-langchain-langgraph",
      ],
    },
    {
      type: "category",
      label: "Deep Dives",
      items: ["concepts/system-architecture"],
    },
    {
      type: "link",
      label: "Examples",
      href: "https://github.com/yiouli/pixie-examples",
    },
    {
      type: "link",
      label: "API Reference",
      href: "pathname:///api/pixie.html",
    },
  ],
};

export default sidebars;
