import { useEffect, useMemo, useState } from "react";
import type { WorkspaceEntry, WorkspaceFile, WorkspaceNode } from "../api/types";

interface Props {
  tree: WorkspaceEntry | null;
  file: WorkspaceFile | null;
  fileLoading: boolean;
  onOpenFile: (path: string) => void;
  onRefresh: () => void;
}

// Parse the tree string into structured rows for rendering with colors.
interface TreeNode {
  indent: number;
  name: string;
  isDir: boolean;
  path: string;
  size?: number | null;
}

const RUNTIME_DIRS = new Set([
  ".venv",
  ".cache",
  ".tmp",
  ".home",
  "__pycache__",
  ".pytest_cache",
]);

function parseTree(treeStr: string, basePath: string): TreeNode[] {
  const lines = treeStr.split("\n").filter((l) => l.trim());
  const nodes: TreeNode[] = [];
  for (const line of lines) {
    // Tree lines look like: "├── src" or "│   └── hello.py"
    const match = line.match(/^([│├└─\s]*)(.+)/);
    if (!match) continue;
    const prefix = match[1];
    const name = match[2].trim();
    const indent = (prefix.match(/│|├|└/g) || []).length;
    const isDir = !name.includes(".") || name.endsWith("/");
    const path = basePath ? `${basePath}/${name}` : name;
    nodes.push({ indent, name, isDir, path });
  }
  return nodes;
}

function filterRuntimeNodes(nodes: WorkspaceNode[], showRuntime: boolean): WorkspaceNode[] {
  if (showRuntime) return nodes;
  return nodes
    .filter((node) => !RUNTIME_DIRS.has(node.name))
    .map((node) => ({
      ...node,
      children: node.children ? filterRuntimeNodes(node.children, showRuntime) : [],
    }));
}

function defaultExpandedPaths(nodes: WorkspaceNode[]): Set<string> {
  const expanded = new Set<string>();
  for (const node of nodes) {
    if (node.type !== "directory") continue;
    if (["workspace", "handoffs", "runs", "uploads", "parsed_requirements"].includes(node.name)) {
      expanded.add(node.path);
    }
  }
  return expanded;
}

function flattenNodes(
  nodes: WorkspaceNode[],
  expanded: Set<string>,
  depth = 0
): TreeNode[] {
  const out: TreeNode[] = [];
  for (const node of nodes) {
    const isDir = node.type === "directory";
    out.push({
      indent: depth,
      name: node.name,
      isDir,
      path: node.path,
      size: node.size,
    });
    if (isDir && expanded.has(node.path)) {
      out.push(...flattenNodes(node.children || [], expanded, depth + 1));
    }
  }
  return out;
}

export function WorkspaceExplorer({ tree, file, fileLoading, onOpenFile, onRefresh }: Props) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [showRuntime, setShowRuntime] = useState(false);

  const nodes = useMemo(() => {
    if (!tree) return [];
    if (tree.nodes && tree.nodes.length > 0) {
      return flattenNodes(filterRuntimeNodes(tree.nodes, showRuntime), expanded);
    }
    return parseTree(tree.tree, tree.path === "." ? "" : tree.path);
  }, [tree, expanded, showRuntime]);

  useEffect(() => {
    if (!tree?.nodes?.length) {
      setExpanded(new Set());
      return;
    }
    setExpanded(defaultExpandedPaths(tree.nodes));
  }, [tree]);

  const handleClick = (node: TreeNode) => {
    if (node.isDir) {
      setExpanded((prev) => {
        const next = new Set(prev);
        if (next.has(node.path)) {
          next.delete(node.path);
        } else {
          next.add(node.path);
        }
        return next;
      });
      return;
    }
    setSelectedPath(node.path);
    onOpenFile(node.path);
  };

  if (!tree) {
    return (
      <div className="panel">
        <div className="empty-state" style={{ fontSize: 12 }}>
          No workspace files yet. Files will appear here as the agent generates them.
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <div className="panel-header">
        <span>{tree.root === "session" ? "Session Files" : "Workspace"}</span>
        <span className="panel-actions">
          <button
            className={`ghost small ${showRuntime ? "active" : ""}`}
            onClick={() => setShowRuntime((v) => !v)}
            title="Show runtime folders"
          >
            Runtime
          </button>
          <button className="ghost small" onClick={onRefresh} title="Refresh">
            Refresh
          </button>
        </span>
      </div>

      <div className="split-vertical">
        <div className="workspace-tree-pane">
          {nodes.length === 0 ? (
            <div style={{ padding: 12, color: "var(--text-muted)", fontSize: 12 }}>
              (empty)
            </div>
          ) : (
            <div className="workspace-tree">
              {nodes.map((node, i) => (
                <div
                  key={i}
                  className={`${node.isDir ? "dir" : "file"} ${
                    selectedPath === node.path ? "selected" : ""
                  }`}
                  style={{
                    paddingLeft: `${node.indent * 16 + 8}px`,
                  }}
                  onClick={() => handleClick(node)}
                  title={node.path}
                >
                  <span className="workspace-node-kind">
                    {node.isDir ? (expanded.has(node.path) ? "[-]" : "[+]") : "   "}
                  </span>
                  <span className="workspace-node-name">{node.name}</span>
                  {!node.isDir && typeof node.size === "number" && (
                    <span className="workspace-node-size">{node.size}b</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div className="file-viewer-header">
            <span>{file?.path || (fileLoading ? "Loading..." : "Select a file")}</span>
            {file && (
              <span style={{ color: "var(--text-muted)" }}>
                {file.size} bytes · {file.encoding}
              </span>
            )}
          </div>
          <div className="file-viewer">
            {file ? file.content : fileLoading ? "Loading..." : ""}
          </div>
        </div>
      </div>
    </div>
  );
}
