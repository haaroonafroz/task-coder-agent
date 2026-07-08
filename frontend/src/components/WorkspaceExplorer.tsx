import { useMemo, useState } from "react";
import type { WorkspaceEntry, WorkspaceFile } from "../api/types";

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
}

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

export function WorkspaceExplorer({ tree, file, fileLoading, onOpenFile, onRefresh }: Props) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const nodes = useMemo(() => {
    if (!tree) return [];
    return parseTree(tree.tree, tree.path === "." ? "" : tree.path);
  }, [tree]);

  const handleClick = (node: TreeNode) => {
    if (!node.isDir) {
      setSelectedPath(node.path);
      onOpenFile(node.path);
    }
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
        <span>Workspace</span>
        <button
          style={{ padding: "2px 8px", fontSize: 11 }}
          onClick={onRefresh}
          title="Refresh"
        >
          ⟳
        </button>
      </div>

      <div className="split-vertical">
        <div style={{ maxHeight: "40%", overflowY: "auto", borderBottom: "1px solid var(--border)" }}>
          {nodes.length === 0 ? (
            <div style={{ padding: 12, color: "var(--text-muted)", fontSize: 12 }}>
              (empty workspace)
            </div>
          ) : (
            <div className="workspace-tree">
              {nodes.map((node, i) => (
                <div
                  key={i}
                  className={node.isDir ? "dir" : "file"}
                  style={{
                    paddingLeft: `${node.indent * 16 + 8}px`,
                    cursor: node.isDir ? "default" : "pointer",
                    background:
                      selectedPath === node.path ? "var(--bg-tertiary)" : "transparent",
                  }}
                  onClick={() => handleClick(node)}
                >
                  {node.isDir ? "📁 " : "📄 "}
                  {node.name}
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
